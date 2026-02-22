/**
 * Authentication error - invalid or missing API key.
 */

import { HowlerError } from "./base.js";

export class AuthenticationError extends HowlerError {
  constructor(message: string = "Authentication failed") {
    super(message, 401);
    this.name = "AuthenticationError";
  }
}
