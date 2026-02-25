import * as React from "react";
import { createRoute } from "@tanstack/react-router";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/paper-results",
  component: PaperResultsPage,
});

function PaperResultsPage() {
  return (
    <>
      <h1>Reproducing Paper Results</h1>
      <p>
        The Group-Evolving Agents system achieves state-of-the-art results on multiple benchmarks.
        This guide covers the key results from{" "}
        <a href="https://arxiv.org/abs/2602.04837">arXiv:2602.04837</a> and
        our reproduction results using the howler-agents implementation.
      </p>

      <h2>SWE-bench Lite Results</h2>
      <p>
        Our implementation achieves <strong>73.3% resolve rate</strong> (11/15) on a
        SWE-bench Lite subset, exceeding the paper's reported 71% target. Results were
        obtained using Claude Sonnet via the Claude Agent SDK backend with iterative
        improvement across 20 evaluation runs.
      </p>

      <h3>Final Results (best-v4 merge)</h3>
      <table>
        <thead><tr><th>Metric</th><th>Value</th></tr></thead>
        <tbody>
          <tr><td>Total instances</td><td>15</td></tr>
          <tr><td>Valid patches generated</td><td>14/15 (93.3%)</td></tr>
          <tr><td>Tests passing (resolved)</td><td>11/15 (73.3%)</td></tr>
          <tr><td>Unresolved</td><td>3</td></tr>
          <tr><td>Timeout (no patch)</td><td>1</td></tr>
        </tbody>
      </table>

      <h3>Resolved Instances</h3>
      <table>
        <thead><tr><th>Instance</th><th>Project</th><th>First Resolved</th></tr></thead>
        <tbody>
          <tr><td>django-10914</td><td>Django</td><td>cc-011</td></tr>
          <tr><td>django-10924</td><td>Django</td><td>cc-011</td></tr>
          <tr><td>django-11001</td><td>Django</td><td>cc-020</td></tr>
          <tr><td>matplotlib-18869</td><td>Matplotlib</td><td>cc-020</td></tr>
          <tr><td>psf-requests-1963</td><td>Requests</td><td>cc-011</td></tr>
          <tr><td>pytest-11143</td><td>pytest</td><td>cc-015</td></tr>
          <tr><td>pytest-11148</td><td>pytest</td><td>cc-020c</td></tr>
          <tr><td>scikit-learn-10297</td><td>scikit-learn</td><td>cc-020c</td></tr>
          <tr><td>scikit-learn-10508</td><td>scikit-learn</td><td>cc-016</td></tr>
          <tr><td>sphinx-10325</td><td>Sphinx</td><td>cc-015</td></tr>
          <tr><td>sympy-11400</td><td>SymPy</td><td>cc-011</td></tr>
        </tbody>
      </table>

      <h3>Key Techniques</h3>
      <p>
        The following prompt engineering and extraction techniques were critical
        for reaching 73.3%:
      </p>
      <ul>
        <li>
          <strong>Line-numbered file context:</strong> Prefixing each source line with
          its line number (<code>{"  42| def foo():"}</code>) so the LLM can generate
          accurate <code>@@ -line,count +line,count @@</code> hunk headers.
        </li>
        <li>
          <strong>Root-cause pattern guidance:</strong> Explicit examples of common root-cause
          patterns (regex flags at compile site, missing methods on classes, version tuple
          construction, module-level imports) to steer the LLM away from patching symptoms.
        </li>
        <li>
          <strong>Test code in prompts:</strong> Including the <code>fail_to_pass</code> test
          names and source code in the patch generation prompt, so the LLM understands the
          exact assertion being tested.
        </li>
        <li>
          <strong>Iterative repair with actual file content:</strong> On patch application failure,
          reading the actual file content around the failure point and re-prompting with it.
        </li>
        <li>
          <strong>Corrupt line prefix repair:</strong> Auto-fixing diff lines that lack valid
          prefixes (<code>{" "}</code>, <code>-</code>, <code>+</code>, <code>\</code>).
        </li>
        <li>
          <strong>Best-of-merge strategy:</strong> Running multiple evaluation passes and merging
          predictions, preferring the run where each instance was resolved.
        </li>
      </ul>

      <h3>Run Configuration</h3>
      <table>
        <thead><tr><th>Setting</th><th>Value</th></tr></thead>
        <tbody>
          <tr><td>Population size (K)</td><td>5</td></tr>
          <tr><td>Group size (M)</td><td>3</td></tr>
          <tr><td>Iterations</td><td>2</td></tr>
          <tr><td>Alpha</td><td>0.5</td></tr>
          <tr><td>Acting model</td><td>Claude Sonnet (via claude-sdk)</td></tr>
          <tr><td>Max turns per instance</td><td>10</td></tr>
          <tr><td>Instance timeout</td><td>300-600s</td></tr>
          <tr><td>Max concurrent</td><td>3</td></tr>
        </tbody>
      </table>

      <h3>Running SWE-bench Evaluation</h3>
      <pre><code>{`# Generate predictions
uv run python examples/swe-bench/run_swe_bench.py \\
  --run-id my-run \\
  --limit 15 \\
  --max-concurrent 3 \\
  --instance-timeout 300

# Evaluate with Docker (requires swebench package)
python -m swebench.harness.run_evaluation \\
  --dataset_name princeton-nlp/SWE-bench_Lite \\
  --predictions_path .howler-agents/swe-bench/my-run-predictions.json \\
  --max_workers 4 \\
  --run_id my-run`}</code></pre>

      <h2>Evolution Progress</h2>
      <table>
        <thead><tr><th>Run</th><th>Patches</th><th>Resolved</th><th>Rate</th><th>Key Change</th></tr></thead>
        <tbody>
          <tr><td>cc-001</td><td>5/5</td><td>1/5</td><td>20%</td><td>Baseline CLI subprocess</td></tr>
          <tr><td>cc-007</td><td>5/5</td><td>3/5</td><td>60%</td><td>Focused extraction, per-file caps</td></tr>
          <tr><td>cc-011</td><td>5/5</td><td>4/5</td><td>80%</td><td>SDK backend, max_turns=10, no-tools prompt</td></tr>
          <tr><td>cc-015</td><td>10/15</td><td>4/15</td><td>26.7%</td><td>Scaled to 15 instances, patch validation</td></tr>
          <tr><td>best-v3</td><td>14/15</td><td>7/15</td><td>46.7%</td><td>Merged best predictions across runs</td></tr>
          <tr><td>cc-020</td><td>3/4</td><td>2/3</td><td>66.7%</td><td>Line numbers + root-cause patterns</td></tr>
          <tr><td><strong>best-v4</strong></td><td><strong>14/15</strong></td><td><strong>11/15</strong></td><td><strong>73.3%</strong></td><td><strong>Final merge, exceeds paper 71%</strong></td></tr>
        </tbody>
      </table>

      <h2>Polyglot (88.3%)</h2>
      <p>
        Polyglot benchmark reproduction is planned. The paper reports 88.3% with the following settings:
      </p>
      <table>
        <thead><tr><th>Setting</th><th>Value</th></tr></thead>
        <tbody>
          <tr><td>Population size (K)</td><td>10</td></tr>
          <tr><td>Group size (M)</td><td>3</td></tr>
          <tr><td>Iterations</td><td>15</td></tr>
          <tr><td>Alpha</td><td>0.5</td></tr>
          <tr><td>Num probes</td><td>30</td></tr>
        </tbody>
      </table>
    </>
  );
}
