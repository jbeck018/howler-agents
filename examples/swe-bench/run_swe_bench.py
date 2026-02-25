"""Run GEA on SWE-bench to evaluate agent performance.

Configuration follows arXiv:2602.04837 Table 3:
- K=50 (population size)
- M=5 (group size)
- alpha=0.7 (performance-weighted -- SWE-bench benefits from exploitation)
- 60 iterations (multi-file coordination requires more generations)
- 30 probe tasks (binary capability vector dimensionality)

SWE-bench tasks involve multi-file edits across real Python repositories.
GEA adapts to this complexity by producing smaller, distributed patches
across more iterations compared to single-file benchmarks.

Usage:
    # Quick test (5 instances, small population):
    python run_swe_bench.py

    # Full paper reproduction (requires API keys and Docker):
    python run_swe_bench.py --limit 0 --population 50 --iterations 60

    # Or use the CLI directly:
    howler-agents swe-bench --limit 5 --model claude-sonnet-4-20250514
"""

import argparse
import asyncio

from howler_agents.benchmarks.swe_bench_runner import SWEBenchEvalRunner


async def main(args: argparse.Namespace) -> None:
    runner = SWEBenchEvalRunner(
        model=args.model,
        dataset=args.dataset,
        population_size=args.population,
        group_size=args.groups,
        num_iterations=args.iterations,
        alpha=args.alpha,
        num_probes=args.probes,
        skip_docker_eval=args.skip_docker_eval,
        max_concurrent=args.max_concurrent,
        instance_timeout_s=args.instance_timeout,
    )

    print("SWE-bench Evaluation -- Howler Agents (GEA)")
    print(f"  Dataset:     {args.dataset}")
    print(f"  Model:       {args.model}")
    print(f"  Population:  {args.population}")
    print(f"  Groups:      {args.groups}")
    print(f"  Iterations:  {args.iterations}")
    print(f"  Alpha:       {args.alpha}")
    print(f"  Parallel:    {args.max_concurrent}, Timeout: {args.instance_timeout}s")
    print(f"  Instances:   {args.limit or 'all'}")
    print()

    report = await runner.run(
        limit=args.limit if args.limit > 0 else None,
        run_id=args.run_id,
    )

    print(report.summary())

    if report.final_results:
        print(f"\nTarget: 71.0% (paper result on SWE-bench Verified)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SWE-bench evaluation with GEA")
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite", help="HuggingFace dataset")
    parser.add_argument("--model", "-m", default="claude-sonnet-4-20250514", help="LiteLLM model")
    parser.add_argument("--population", "-p", type=int, default=6, help="Population size (K)")
    parser.add_argument("--groups", "-g", type=int, default=2, help="Group size (M)")
    parser.add_argument("--iterations", "-n", type=int, default=3, help="Evolution generations")
    parser.add_argument("--alpha", "-a", type=float, default=0.7, help="Performance vs novelty")
    parser.add_argument("--probes", type=int, default=15, help="Number of probe tasks")
    parser.add_argument("--limit", type=int, default=5, help="Number of instances (0=all)")
    parser.add_argument("--run-id", default=None, help="Custom run ID")
    parser.add_argument("--skip-docker-eval", action="store_true", help="Skip Docker evaluation")
    parser.add_argument("--max-concurrent", type=int, default=3, help="Max parallel prediction workers")
    parser.add_argument("--instance-timeout", type=float, default=300.0, help="Per-instance timeout (seconds)")
    asyncio.run(main(parser.parse_args()))
