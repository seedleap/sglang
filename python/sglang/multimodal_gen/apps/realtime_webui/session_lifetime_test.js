const assert = require("assert");

const { SessionLifetimeGuard } = require("./session_lifetime.js");

let scheduled = null;
let cleared = [];
let expirations = 0;
const guard = new SessionLifetimeGuard({
  durationMs: 45_000,
  setTimer: (callback, delay) => {
    scheduled = { callback, delay, id: 7 };
    return 7;
  },
  clearTimer: (id) => cleared.push(id),
  onExpire: () => { expirations += 1; },
});

guard.start();
assert.equal(scheduled.delay, 45_000);
scheduled.callback();
assert.equal(expirations, 1, "guard should expire one active session");

guard.start();
guard.cancel();
assert.ok(cleared.includes(7), "cancel should clear the active timer");
scheduled.callback();
assert.equal(expirations, 1, "a cancelled generation must not expire a replacement session");

console.log("session lifetime guard ok");
