import { defineConfig } from "orval";

export default defineConfig({
  howlerApi: {
    input: {
      target: "http://localhost:8080/openapi.json",
    },
    output: {
      target: "./src/generated/api.ts",
      client: "react-query",
      mode: "tags-split",
      override: {
        mutator: {
          path: "./src/lib/api-fetcher.ts",
          name: "customFetch",
        },
      },
    },
  },
});
