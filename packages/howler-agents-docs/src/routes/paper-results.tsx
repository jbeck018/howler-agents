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
        This guide helps reproduce the key results from{" "}
        <a href="https://arxiv.org/abs/2602.04837">arXiv:2602.04837</a>.
      </p>

      <h2>SWE-bench (71%)</h2>
      <table>
        <thead><tr><th>Setting</th><th>Value</th></tr></thead>
        <tbody>
          <tr><td>Population size (K)</td><td>10</td></tr>
          <tr><td>Group size (M)</td><td>3</td></tr>
          <tr><td>Iterations</td><td>10</td></tr>
          <tr><td>Alpha</td><td>0.5</td></tr>
          <tr><td>Acting model</td><td>Claude Sonnet</td></tr>
          <tr><td>Evolving model</td><td>Claude Sonnet</td></tr>
        </tbody>
      </table>

      <h2>Polyglot (88.3%)</h2>
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
