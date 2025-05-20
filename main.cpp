// MNIST model trainig example
#include <iostream>
#include <torch/csrc/api/include/torch/torch.h>
#include <torch/csrc/distributed/c10d/Types.hpp>

#ifndef SERIALISED_VERSION
# include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
# include <torch/csrc/distributed/c10d/ProcessGroupMPI.hpp>
# include <torch/csrc/distributed/c10d/Work.hpp>

# define RUN_MODE_STRING "DISTRIBUTED"
#else
# define RUN_MODE_STRING "SERIALISED "
#endif


// Configurable options
#define PROVIDE_TIMING
#define MEASURE_TORCH_BUILTIN_OVERLAP


//


#if (defined(PROVIDE_TIMING))
# include <chrono>
using namespace std::chrono_literals;
#endif


#if (defined(MEASURE_TORCH_BUILTIN_OVERLAP) && !(defined(SERIALISED_VERSION)))
# define __DO_MEASURE_TORCH_BUILTIN_OVERLAP
#endif


#define TENSOR_NORMALISE_OP_CST_1 0.1307
#define TENSOR_NORMALISE_OP_CST_2 0.3081


#define CHRONO_SECONDS_TO_DOUBLE(t)                                                                \
    (double)((double)std::chrono::duration_cast<std::chrono::seconds>(t).count() +                 \
             1e-3 * (double)std::chrono::duration_cast<std::chrono::milliseconds>(t % 1s).count())

//


// Define a Convolutional Module
struct Model : torch::nn::Module {
    Model()
        : conv1(torch::nn::Conv2dOptions(1, 10, 5)), conv2(torch::nn::Conv2dOptions(10, 20, 5)),
          fc1(320, 50), fc2(50, 10)
    {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv2_drop", conv2_drop);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
        x = torch::relu(torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
        x = x.view({-1, 320});
        x = torch::relu(fc1->forward(x));
        x = torch::dropout(x, 0.5, is_training());
        x = fc2->forward(x);
        return torch::log_softmax(x, 1);
    }

    torch::nn::Conv2d    conv1;
    torch::nn::Conv2d    conv2;
    torch::nn::Dropout2d conv2_drop;
    torch::nn::Linear    fc1;
    torch::nn::Linear    fc2;
};


#ifdef __DO_MEASURE_TORCH_BUILTIN_OVERLAP
# define DIM 25

static inline void dummy_cpu_compute(void)
{
    static volatile float a[DIM][DIM];
    static volatile float x[DIM];
    static volatile float y[DIM];

    int i = 0, j = 0;
    for (i = 0; i < DIM; i++)
        for (j = 0; j < DIM; j++)
            x[i] = x[i] + a[i][j] * a[j][i] + y[j];
}

static inline void do_cpu_computation(double target_seconds)
{
    double t1 = 0.0, t2 = 0.0;
    double time_elapsed = 0.0;
    while (time_elapsed < target_seconds) {
        t1 = MPI_Wtime();
        dummy_cpu_compute();
        t2 = MPI_Wtime();
        time_elapsed += (t2 - t1);
    }
}


// The closer the parallel time to the serial, the less the ovelap.
static inline double compute_overlap(double serial, double parallel)
{
    double overlap = 0.0;
    overlap        = serial - parallel;
    if (overlap < 0.0) {
        return 0.0;
    }
    return (overlap / serial) * 100.0;
}
#endif


int main(int argc, char *argv[])
{
#ifndef SERIALISED_VERSION
    // Creating MPI Process Group
    auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();

    // Retrieving MPI environment variables
    const auto numranks = pg->getSize();
    const auto rank     = pg->getRank();
#else
    constexpr auto numranks = 1;
    constexpr auto rank     = 0;
#endif
    if (0 == rank) {
        fprintf(stdout, "\n ****  " RUN_MODE_STRING "  MNIST  toy model  ****\n\n");
    }

    // Config data
    const auto                   total_batch_size = 64 * numranks;
    constexpr size_t             num_epochs       = 10;
    constexpr auto               learning_rate    = 1e-2;
    const c10d::AllreduceOptions opts             = c10d::AllreduceOptions();

    torch::manual_seed(0);


    // Timings
#if (defined(PROVIDE_TIMING))
    auto hr_clock        = std::chrono::high_resolution_clock();
    auto time_init       = hr_clock.now();
    auto time_train_init = time_init;
    auto time_train_end  = time_train_init;
    auto time_test_init  = time_init;

# ifndef SERIALISED_VERSION
#  ifdef __DO_MEASURE_TORCH_BUILTIN_OVERLAP
    std::chrono::duration<double> duration_allreduce_plus_wait_serial(0.0);
    std::chrono::duration<double> duration_allreduce_plus_wait(0.0);
    std::chrono::duration<double> duration_allreduce(0.0);
    std::chrono::duration<double> duration_wait(0.0);

    std::vector<std::vector<double>> overlaps(num_epochs);
#  endif
# endif
#endif


    // ***********
    // TRAINING
    // ***********
    // Read train dataset
    const char *kDataRoot     = "/home/bsc/bsc488161/mpi_offload/tests/mnist/dataset/dataset/";
    auto        train_dataset = torch::data::datasets::MNIST(kDataRoot)
                             .map(torch::data::transforms::Normalize<>(TENSOR_NORMALISE_OP_CST_1,
                                                                       TENSOR_NORMALISE_OP_CST_2))
                             .map(torch::data::transforms::Stack<>());

    // Generate dataloader
#ifndef SERIALISED_VERSION
    auto data_sampler = torch::data::samplers::DistributedRandomSampler(
        train_dataset.size().value(), numranks, rank, false);
#else
    auto data_sampler = torch::data::samplers::RandomSampler(train_dataset.size().value());
#endif
    auto num_train_samples_per_proc = train_dataset.size().value() / numranks;
    auto batch_size_per_proc =
        total_batch_size / numranks; // effective batch size in each processor
    auto data_loader =
        torch::data::make_data_loader(std::move(train_dataset), data_sampler, batch_size_per_proc);

    // Create model instance
    auto model = std::make_shared<Model>();

    // Create optimizer instance
    torch::optim::SGD optimizer(model->parameters(), learning_rate);

    size_t n_train_batches;
    for (size_t epoch = 1; epoch <= num_epochs; ++epoch) {

        size_t num_correct = 0;

#ifdef __DO_MEASURE_TORCH_BUILTIN_OVERLAP
        overlaps[epoch - 1].reserve(1000);
#endif

        n_train_batches = 0;
        for (auto &batch : *data_loader) {
            auto ip = batch.data;
            auto op = batch.target.squeeze();

            ++n_train_batches;

            // convert to required formats
            ip = ip.to(torch::kF32);
            op = op.to(torch::kLong);

            // Reset gradients
            model->zero_grad();

            // Execute forward pass
            auto prediction = model->forward(ip);

            auto loss = torch::nll_loss(torch::log_softmax(prediction, 1), op);

            // Backpropagation
            loss.backward();

#ifndef SERIALISED_VERSION
# if (defined(PROVIDE_TIMING))
            auto ts = hr_clock.now();
# endif

# ifdef __DO_MEASURE_TORCH_BUILTIN_OVERLAP
            for (auto &param : model->named_parameters()) {
                std::vector<torch::Tensor> tmp = {param.value().grad()};

                auto work = pg->allreduce(tmp, opts);
                work->wait();
            }
            // NOTE: do not increment this since batch size might vary
            duration_allreduce_plus_wait_serial = hr_clock.now() - ts;

            // NOTE: also duplicate averaging, to not alter next reduction operation
            for (auto &param : model->named_parameters()) {
                param.value().grad().data() = param.value().grad().data() / numranks;
            }

            // Restore start time
            ts = hr_clock.now();
# endif

            // Averaging the gradients of the parameters in all the processors
            // Note: This may lag behind DistributedDataParallel (DDP) in performance
            // since this synchronizes parameters after backward pass while DDP
            // overlaps synchronizing parameters and computing gradients in backward pass.
            std::vector<c10::intrusive_ptr<::c10d::Work>> works;
            for (auto &param : model->named_parameters()) {
                std::vector<torch::Tensor> tmp = {param.value().grad()};
# ifdef __DO_MEASURE_TORCH_BUILTIN_OVERLAP
                auto ti = hr_clock.now();
# endif
                auto work = pg->allreduce(tmp, opts);
# ifdef __DO_MEASURE_TORCH_BUILTIN_OVERLAP
                duration_allreduce += hr_clock.now() - ti;
# endif
                works.push_back(std::move(work));
            }

# ifdef __DO_MEASURE_TORCH_BUILTIN_OVERLAP
            // NOTE: compute time does not include the time need for the Allreduce calls.
            auto duration_wait_serial = (duration_allreduce_plus_wait_serial - duration_allreduce);
            do_cpu_computation(duration_wait_serial.count());

            auto ti = hr_clock.now();
# endif

            for (auto &work : works) {
                try {
                    work->wait();
                } catch (const std::exception &ex) {
                    std::cerr << "Exception received: " << ex.what() << std::endl;
                    pg->abort();
                }
            }

# ifdef __DO_MEASURE_TORCH_BUILTIN_OVERLAP
            auto te                      = hr_clock.now();
            duration_wait                = te - ti;
            duration_allreduce_plus_wait = te - ts;

            overlaps[epoch - 1].emplace_back(
                compute_overlap(duration_wait_serial.count(), duration_wait.count()));
# endif

            for (auto &param : model->named_parameters()) {
                param.value().grad().data() = param.value().grad().data() / numranks;
            }
#endif // SERIALISED_VERSION

            // Update parameters
            optimizer.step();

            auto guess = prediction.argmax(1);
            num_correct += torch::sum(guess.eq_(op)).item<int64_t>();
        } // end batch loader

        auto accuracy = 100.0 * num_correct / num_train_samples_per_proc;
        std::cout << "Accuracy "
#ifndef SERIALISED_VERSION
                  << "in rank " << rank
#endif
                  << " in epoch " << epoch << " - " << accuracy << std::endl;
    } // end epoch

#if (defined(PROVIDE_TIMING))
    if (0 == rank) {
        time_train_end = hr_clock.now();

        time_test_init = time_train_end;
    }
#endif


    // **********************
    // TRAINING (ONLY RANK 0)
    // **********************
    size_t n_test_batches;
    if (rank == 0) {
        auto test_dataset =
            torch::data::datasets::MNIST(kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
                .map(torch::data::transforms::Normalize<>(TENSOR_NORMALISE_OP_CST_1,
                                                          TENSOR_NORMALISE_OP_CST_2))
                .map(torch::data::transforms::Stack<>());

        auto num_test_samples = test_dataset.size().value();
        auto test_loader = torch::data::make_data_loader(std::move(test_dataset), num_test_samples);

        model->eval(); // enable eval mode to prevent backprop

        size_t num_correct = 0;

        n_test_batches = 0;
        for (auto &batch : *test_loader) {
            auto ip = batch.data;
            auto op = batch.target.squeeze();

            ++n_test_batches;

            // convert to required format
            ip = ip.to(torch::kF32);
            op = op.to(torch::kLong);

            auto prediction = model->forward(ip);
            auto loss       = torch::nll_loss(torch::log_softmax(prediction, 1), op);
            auto guess      = prediction.argmax(1);
            num_correct += torch::sum(guess.eq_(op)).item<int64_t>();

            std::cout << "Test loss - " << loss.item<float>() << std::endl;
        } // end test loader

        std::cout << "Num correct - " << num_correct << std::endl;
        std::cout << "Test Accuracy - " << 100.0 * num_correct / num_test_samples << std::endl;
    } // end rank 0


#if (defined(PROVIDE_TIMING))
    if (0 == rank) {
        auto test_time  = hr_clock.now() - time_test_init;
        auto train_time = time_train_end - time_train_init;

        fprintf(stdout, "\nTiming report for  %ld  epochs:\n", num_epochs);
        fprintf(stdout, "\t%-12s %7.3lf s  (%5ld batches)\n",
                "training:", CHRONO_SECONDS_TO_DOUBLE(train_time), n_train_batches);
        fprintf(stdout, "\t%-12s %7.3lf s  (%5ld batches)\n",
                "test:", CHRONO_SECONDS_TO_DOUBLE(test_time), n_test_batches);
# ifdef __DO_MEASURE_TORCH_BUILTIN_OVERLAP
        double avg_overlap = 0.0;
        for (size_t i = 0; i < num_epochs; ++i) {
            for (size_t j = 0; j < n_train_batches; ++j) {
                avg_overlap += overlaps[i][j];
            }
        }
        avg_overlap /= (num_epochs * n_train_batches);
        fprintf(stdout, "\t%-12s %7.3lf %%\n", "overlap:", avg_overlap);
# endif
    }
#endif
}
