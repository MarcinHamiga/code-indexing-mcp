// Pins current (buggy) extraction for backlog defects E4, E5, E6, E9, E11.
//
// Task 0.1 freezes the wrong output rather than fixing it -- see the
// hardening plan, Phase 0.

import "./polyfill"; // E9: side-effect import emits no reference at all
import { Widget } from "./widget";

const lazy = require("./lazy"); // E9: require() call keeps no module_path
const dynamic = import("./dynamic"); // E9: dynamic import() call keeps no module_path

@sealed // E6: decorators produce zero references in the JS/TS family
class Service {
  @readonly
  handle() {
    return this.value;
  }
}

function summarize(...items) {
  return items.length;
}

const config = { TIMEOUT: 30 };

function touch(target) {
  target.TIMEOUT = 5; // E5: member-expression write is swallowed
  return target.TIMEOUT; // E5: member-expression read never appears
}

function build(onSave) {
  return { onSave }; // E5: shorthand_property_identifier is not identifier, missed
}

function iterate(items) {
  for (const item of items) {
    // E11: for...of binding registers as a spurious `read`
    console.log(item);
  }
}

new Widget; // E4: constructor call without an arguments node produces nothing

const tagged = gql`
  query { widgets }
`; // E4: tagged template has no `arguments` node

export * from "./x"; // E3: barrel re-export produces no reference at all
export * as ns from "./x"; // E3: namespaced barrel re-export loses the module path
