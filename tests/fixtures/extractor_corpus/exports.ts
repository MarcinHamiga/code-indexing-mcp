export const VERSION = "1.0.0";

export interface Options {
  retries: number;
}

export function configure(options: Options): Options {
  function normalize(value: number): number {
    return Math.max(0, value);
  }
  return { retries: normalize(options.retries) };
}

export default class Client {
  constructor(private readonly options: Options) {}

  async send(payload: string): Promise<string> {
    return payload;
  }
}

const anonymous = () => VERSION;
export { anonymous };
