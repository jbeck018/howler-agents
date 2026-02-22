/**
 * Base error class for Howler Agents SDK.
 */

export class HowlerError extends Error {
  public readonly statusCode?: number;

  constructor(message: string, statusCode?: number) {
    super(message);
    this.name = "HowlerError";
    this.statusCode = statusCode;
  }
}
