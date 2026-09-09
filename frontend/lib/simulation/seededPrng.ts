/**
 * Horizon 2 Foundation: Seeded Pseudo-Random Number Generator (XorShift32)
 *
 * Strictly adheres to INV-OI54 (Simulation Determinism):
 * 100 Replays with identical seed generate identical pseudo-random stream
 * and identical downstream simulation hashes. Zero Math.random() usage.
 */

export class SeededPrng {
  private state: number;
  private initialSeed: number;

  constructor(seed = 123456789) {
    // Avoid 0 as XorShift32 seed
    this.state = (seed >>> 0) || 1;
    this.initialSeed = this.state;
  }

  public getSeed(): number {
    return this.initialSeed;
  }

  /**
   * Fast 32-bit XorShift pseudo-random number generator
   * Period: 2^32 - 1
   */
  public next(): number {
    let x = this.state;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.state = x >>> 0;
    return this.state / 4294967296;
  }

  /**
   * Uniform floating-point number in [min, max)
   */
  public uniform(min: number, max: number): number {
    return min + this.next() * (max - min);
  }

  /**
   * Triangular distribution sampling for asymmetric risk and impact parameters
   * @param min Minimum bound
   * @param mode Most likely value (peak)
   * @param max Maximum bound
   */
  public triangular(min: number, mode: number, max: number): number {
    const u = this.next();
    const c = (mode - min) / (max - min);

    if (u < c) {
      return min + Math.sqrt(u * (max - min) * (mode - min));
    }
    return max - Math.sqrt((1 - u) * (max - min) * (max - mode));
  }

  /**
   * Reset generator to initial seed
   */
  public reset(): void {
    this.state = this.initialSeed;
  }
}
