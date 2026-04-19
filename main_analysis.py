"""
Llama 3.1 8B Q/K/Attention Matrix Analysis & Visualization

Author: Zheng Liu
Date: 2025/06/19
"""
import argparse
import torch

from llama_3_1_8B_Instruct_Test.llama_test_utils.model_loader import ModelLoader
from llama_test_utils.data_loader import DataLoader
from llama_test_utils.qka_collector import QKAttentionCollector
from llama_test_utils.analyzer import QKAAnalyzer

def main(benchmark="longbench", run_high_score=False, run_heatmap=False, run_sparsity=False, run_qk_matrix=False, run_block_sim=False, run_grid_mask=False, run_grid_eff=False):
    """Main execution function for Q/K/Attention analysis pipeline."""
    print("Llama 3.1 8B Q/K/Attention Matrix Analysis")
    print("=" * 40)

    try:
        # Step 1: Load model
        print("Step 1: Loading model...")
        model_loader = ModelLoader()

        # Step 2: Load data
        print("\nStep 2: Loading data...")
        data_loader = DataLoader()
        text = data_loader.load_data('longbench')

        # Step 3: Collect Q/K matrices
        print("\nStep 3: Collecting Q/K and Attention matrices...")
        collector = QKAttentionCollector(model_loader)
        q_matrices, k_matrices, attention_matrices = collector.collect_qk_attention_matrices(text, max_length=2048)

        # Step 4: Visualize matrices
        print("\nStep 4: Visualizing matrices...")
        visualizer = QKAAnalyzer()

        if run_grid_eff:
            print("Running Grid Attention Effectiveness Visualization...")
            visualizer.visualize_grid_attention_effectiveness(q_matrices, k_matrices, attention_matrices, target_mass=0.97, block_size=128, grid_size=8, output_dir="grid_attention_effectiveness_results")

        if run_heatmap:
            print("Running Layer-Head Sparsity Heatmap...")
            visualizer.visualize_layer_head_sparsity_heatmap(attention_matrices, target_mass=0.97, output_dir="layer_head_sparsity_heatmap")

        if run_high_score:
            print("Running High Score Analysis...")
            visualizer.visualize_high_score_parts_in_post_softmax_matrix(attention_matrices, target_mass= 0.95, output_dir="high_score_in_post_softmax_results")

        if run_sparsity:
            print("Running Sparsity vs Mass Analysis...")
            visualizer.visualize_sparsity_vs_mass(attention_matrices, output_dir="sparsity_analysis")

        if run_qk_matrix:
            print("Running Q/K Matrix Visualization...")
            visualizer.visualize_qk_matrix(q_matrices, k_matrices, output_dir="qk_matrix_visualization")

        if run_block_sim:
            print("Running Block Similarity Analysis...")
            visualizer.visualize_block_similarity(q_matrices, k_matrices, block_sizes=[16, 32, 64], output_dir="block_similarity_analysis")

        if run_grid_mask:
            print("Running Grid Attention Block Sparse Mask Visualization...")
            visualizer.visualize_grid_attention_block_sparse_mask(q_matrices, k_matrices, attention_matrices, target_mass=0.8, block_size=128, output_dir="grid_attention_block_sparse_mask_results")

        if not any([run_high_score, run_heatmap, run_sparsity, run_qk_matrix, run_block_sim, run_grid_mask, run_grid_eff]):
            print("No analysis selected. Use flags: --high-score, --heatmap, --sparsity, --qk-matrix, --block-sim, --grid-mask, --grid-eff")

        print("\nAnalysis completed!")

    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        # Cleanup
        if "model_loader" in locals():
            model_loader.cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama 3.1 8B Q/K/Attention Matrix Analysis")
    parser.add_argument("--benchmark", type=str, default="longbench", choices=["longbench", "custom"], help="Benchmark dataset to use")
    parser.add_argument("--high-score", action="store_true", help="Run high score parts visualization")
    parser.add_argument("--heatmap", action="store_true", help="Run layer-head sparsity heatmap visualization")
    parser.add_argument("--sparsity", action="store_true", help="Run sparsity vs mass visualization")
    parser.add_argument("--qk-matrix", action="store_true", help="Run Q/K matrix visualization")
    parser.add_argument("--block-sim", action="store_true", help="Run block similarity analysis")
    parser.add_argument("--grid-mask", action="store_true", help="Run grid attention block sparse mask visualization")
    parser.add_argument("--grid-eff", action="store_true", help="Run grid attention effectiveness visualization")

    args = parser.parse_args()

    main(benchmark=args.benchmark, run_high_score=args.high_score, run_heatmap=args.heatmap, run_sparsity=args.sparsity, run_qk_matrix=args.qk_matrix, run_block_sim=args.block_sim, run_grid_mask=args.grid_mask, run_grid_eff=args.grid_eff)