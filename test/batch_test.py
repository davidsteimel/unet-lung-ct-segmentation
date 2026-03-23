import os
import sys
import subprocess
import argparse
from pathlib import Path

TEST_CONFIGS = [
    (256, "fixed"),
    (256, "flex"),
    (384, "fixed"),
    (384, "flex"),
    (512, "fixed"),
    (512, "flex"),
    (768, "fixed"),
    (768, "flex"),
    (1024, "fixed"),
    (1024, "flex"),
]

def run_test(config_path, test_images, test_masks, seed=42, n_visual=6, threshold=0.5, output_dir=None):
    script_dir = Path(__file__).parent
    test_script = script_dir / "test_unet.py"
    
    cmd = [
        "python", "-u", str(test_script),
        "--config", config_path,
        "--test_images", test_images,
        "--test_masks", test_masks,
        "--seed", str(seed),
        "--n_visual", str(n_visual),
        "--threshold", str(threshold),
    ]
    
    if output_dir:
        cmd.extend(["--output_dir", output_dir])

    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        print(f"Test failed with exit code {result.returncode}")
        return False
    
    print(f"Test completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Batch test script for all resolutions and kernel types"
    )
    parser.add_argument(
        "--configs_dir",
        default="experiments/unettest",
        help="Base directory containing test configs"
    )
    parser.add_argument(
        "--test_data_base",
        default="data_processed",
        help="Base directory containing test data"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=44,
        help="Seed for reproducible sample selection (same across all tests)"
    )
    parser.add_argument(
        "--n_visual",
        type=int,
        default=6,
        help="Number of samples to visualize"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Segmentation threshold"
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=[256, 384, 512, 768, 1024],
        help="Resolutions to test (default: all)"
    )
    parser.add_argument(
        "--kernels",
        nargs="+",
        choices=["fixed", "flex"],
        default=["fixed", "flex"],
        help="Kernel types to test (default: both)"
    )
    parser.add_argument(
        "--skip-on-error",
        action="store_true",
        help="Continue with next test if one fails"
    )
    parser.add_argument(
        "--output_base",
        default="test_results_unet",
        help="Base directory for test results"
    )
    
    args = parser.parse_args()
    
    # Filter configurations based on arguments
    configs_to_test = [
        (res, kernel) 
        for res, kernel in TEST_CONFIGS 
        if res in args.resolutions and kernel in args.kernels
    ]
    
    if not configs_to_test:
        print("No configurations match the specified filters")
        sys.exit(1)
    
    results = {}
    failed_tests = []
    
    for i, (res, kernel) in enumerate(configs_to_test, 1):
        config_name = f"{res}_{kernel}/config.yaml"
        config_path = os.path.join(args.configs_dir, config_name)
        
        test_images = os.path.join(args.test_data_base, str(res), "test", "image")
        test_masks = os.path.join(args.test_data_base, str(res), "test", "mask")
        
        output_dir = os.path.join(args.output_base, f"{res}_{kernel}")

        if not os.path.exists(config_path):
            print(f"Config not found: {config_path}")
            print(f"Skipping {res}_{kernel}")
            results[f"{res}_{kernel}"] = "SKIPPED "
            continue
        
        if not os.path.exists(test_images):
            print(f"Test images not found: {test_images}")
            print(f"Skipping {res}_{kernel}")
            results[f"{res}_{kernel}"] = "SKIPPED"
            continue
        
        if not os.path.exists(test_masks):
            print(f"Test masks not found: {test_masks}")
            print(f"Skipping {res}_{kernel}")
            results[f"{res}_{kernel}"] = "SKIPPED"
            continue
        
        # Run test
        success = run_test(
            config_path,
            test_images,
            test_masks,
            seed=args.seed,
            n_visual=args.n_visual,
            threshold=args.threshold,
            output_dir=output_dir
        )
        
        if success:
            results[f"{res}_{kernel}"] = "SUCCESS"
        else:
            results[f"{res}_{kernel}"] = "FAILED"
            failed_tests.append(f"{res}_{kernel}")
            
            if not args.skip_on_error:
                print(f"Stopping due to failed test: {res}_{kernel}")
                break
    
    successful = sum(1 for s in results.values() if s == "SUCCESS")
    skipped = sum(1 for s in results.values() if s.startswith("SKIPPED"))
    
    if results:
        print("\n  Detailed Results:")
        for config, status in results.items():
            symbol = "✓" if status == "SUCCESS" else ("✗" if status == "FAILED" else "⊘")
            print(f"    {symbol} {config:20s} → {status}")
        print(f"{'='*80}\n")

    if failed_tests:
        print(f"{len(failed_tests)} test(s) failed: {', '.join(failed_tests)}")
        sys.exit(1)
    else:
        print(f"All {successful} tests completed successfully!")
        if skipped > 0:
            print(f"  ({skipped} tests skipped)")
        sys.exit(0)


if __name__ == "__main__":
    main()