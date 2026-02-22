/**
 * Resource not found error.
 */

import { HowlerError } from "./base.js";

export class NotFoundError extends HowlerError {
  constructor(message: string) {
    super(message, 404);
    this.name = "NotFoundError";
  }
}
