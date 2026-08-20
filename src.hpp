#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  const size_t num_heads = keys.size();

  // Precompute K, K_T, V once (they are same for all queries)
  // Step 1: Concatenate keys into K (num_heads x 512)
  Matrix *K = matrix_memory_allocator.Allocate("K");
  gpu_sim.Copy(keys[0], K, Position::kInGpuHbm);
  gpu_sim.Run(false, &matrix_memory_allocator);
  for (size_t h = 1; h < num_heads; ++h) {
    Matrix *temp_K = matrix_memory_allocator.Allocate("temp_K_" + std::to_string(h));
    gpu_sim.Concat(K, keys[h], temp_K, 0, Position::kInGpuHbm);
    gpu_sim.Run(false, &matrix_memory_allocator);
    K = temp_K;
  }

  // Step 2: Transpose K to K_T (512 x num_heads)
  Matrix *K_T = matrix_memory_allocator.Allocate("K_T");
  gpu_sim.Copy(K, K_T, Position::kInGpuHbm);
  gpu_sim.Run(false, &matrix_memory_allocator);
  gpu_sim.Transpose(K_T, Position::kInGpuHbm);
  gpu_sim.Run(false, &matrix_memory_allocator);

  // Step3: Concatenate values into V (num_heads x 512)
  Matrix *V = matrix_memory_allocator.Allocate("V");
  gpu_sim.Copy(values[0], V, Position::kInGpuHbm);
  gpu_sim.Run(false, &matrix_memory_allocator);
  for (size_t h = 1; h < num_heads; ++h) {
    Matrix *temp_V = matrix_memory_allocator.Allocate("temp_V_" + std::to_string(h));
    gpu_sim.Concat(V, values[h], temp_V, 0, Position::kInGpuHbm);
    gpu_sim.Run(false, &matrix_memory_allocator);
    V = temp_V;
  }

  // Process all queries
  for (size_t q_idx = 0; q_idx < keys.size(); ++q_idx) {
    auto current_query = rater.GetNextQuery();
    const size_t m = current_query->GetRowNum();
    std::cerr << "=== Query " << q_idx << ": m=" << m << " ===" << std::endl;

    // Step4: Move Q, K_T, V to SRAM (for this query)
    Matrix *Q_sram = matrix_memory_allocator.Allocate("Q_sram_" + std::to_string(q_idx));
    gpu_sim.Copy(current_query, Q_sram, Position::kInGpuHbm);
    gpu_sim.Run(false, &matrix_memory_allocator);
    gpu_sim.MoveMatrixToSharedMem(Q_sram);
    gpu_sim.Run(false, &matrix_memory_allocator);

    Matrix *K_T_sram = matrix_memory_allocator.Allocate("K_T_sram_" + std::to_string(q_idx));
    gpu_sim.Copy(K_T, K_T_sram, Position::kInGpuHbm);
    gpu_sim.Run(false, &matrix_memory_allocator);
    gpu_sim.MoveMatrixToSharedMem(K_T_sram);
    gpu_sim.Run(false, &matrix_memory_allocator);

    Matrix *V_sram = matrix_memory_allocator.Allocate("V_sram_" + std::to_string(q_idx));
    gpu_sim.Copy(V, V_sram, Position::kInGpuHbm);
    gpu_sim.Run(false, &matrix_memory_allocator);
    gpu_sim.MoveMatrixToSharedMem(V_sram);
    gpu_sim.Run(false, &matrix_memory_allocator);

    // Step5: Compute Q @ K_T
    Matrix *QK_T_sram = matrix_memory_allocator.Allocate("QK_T_sram_" + std::to_string(q_idx));
    gpu_sim.MatMul(Q_sram, K_T_sram, QK_T_sram);
    gpu_sim.Run(false, &matrix_memory_allocator);

    // Step6: Apply row-wise softmax
    Matrix *S_sram = nullptr;
    for (size_t row = 0; row < m; ++row) {
      Matrix *row_vec = matrix_memory_allocator.Allocate("row_" + std::to_string(q_idx) + "_" + std::to_string(row));
      gpu_sim.GetRow(QK_T_sram, row, row_vec, Position::kInSharedMemory);
      gpu_sim.Run(false, &matrix_memory_allocator);

      Matrix *exp_row = matrix_memory_allocator.Allocate("exp_row_" + std::to_string(q_idx) + "_" + std::to_string(row));
      gpu_sim.MatExp(row_vec, exp_row);
      gpu_sim.Run(false, &matrix_memory_allocator);

      Matrix *sum_exp = matrix_memory_allocator.Allocate("sum_exp_" + std::to_string(q_idx) + "_" + std::to_string(row));
      gpu_sim.Sum(exp_row, sum_exp);
      gpu_sim.Run(false, &matrix_memory_allocator);

      Matrix *softmax_row = matrix_memory_allocator.Allocate("softmax_row_" + std::to_string(q_idx) + "_" + std::to_string(row));
      gpu_sim.MatDiv(exp_row, sum_exp, softmax_row);
      gpu_sim.Run(false, &matrix_memory_allocator);

      if (S_sram == nullptr) {
        S_sram = softmax_row;
      } else {
        Matrix *temp_S = matrix_memory_allocator.Allocate("temp_S_" + std::to_string(q_idx) + "_" + std::to_string(row));
        gpu_sim.Concat(S_sram, softmax_row, temp_S, 0, Position::kInSharedMemory);
        gpu_sim.Run(false, &matrix_memory_allocator);
        S_sram = temp_S;
      }
    }

    // Step7: Compute S @ V
    Matrix *answer_sram = matrix_memory_allocator.Allocate("answer_sram_" + std::to_string(q_idx));
    gpu_sim.MatMul(S_sram, V_sram, answer_sram);
    gpu_sim.Run(false, &matrix_memory_allocator);

    // Step8: Move to HBM and commit
    gpu_sim.MoveMatrixToGpuHbm(answer_sram);
    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*answer_sram);

    std::cerr << "=== Query " << q_idx << " committed ===" << std::endl;
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
