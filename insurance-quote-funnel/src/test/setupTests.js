import "@testing-library/jest-dom";

if (!globalThis.crypto) {
  globalThis.crypto = {};
}

if (!globalThis.crypto.randomUUID) {
  globalThis.crypto.randomUUID = () => "test-uuid";
}
