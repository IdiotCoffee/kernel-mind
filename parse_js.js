const fs = require("fs");
const parser = require("@babel/parser");

const filePath = process.argv[2]; // first CLI argument

try {
  const src = fs.readFileSync(filePath, "utf8");

  const ast = parser.parse(src, {
    sourceType: "unambiguous",
    plugins: [
      "typescript",
      "jsx",
      "classProperties",
      "classPrivateProperties",
      "decorators-legacy",
      "dynamicImport",
      "optionalChaining",
      "nullishCoalescingOperator",
      "topLevelAwait",
    ],
  });

  console.log(JSON.stringify({ ok: true, ast }, null, 2));
} catch (err) {
  console.log(JSON.stringify({ ok: false, error: err.message }));
}
