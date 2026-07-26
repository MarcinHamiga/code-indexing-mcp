const CONSTANT = 42;

function classic(value) {
  return value + CONSTANT;
}

const arrow = (value) => classic(value);

class Legacy {
  constructor(seed) {
    this.seed = seed;
  }

  compute() {
    return arrow(this.seed);
  }
}

module.exports = { classic, arrow, Legacy };
