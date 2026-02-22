/**
 * Connection error - service unreachable.
 */

import { HowlerError } from "./base.js";

export class ConnectionError extends HowlerError {
  constructor(message: string) {
    super(message, 503);
    this.name = "ConnectionError";
  }
}
