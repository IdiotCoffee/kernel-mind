const fs = require("fs");
const parser = require("@babel/parser");
const traverse = require("@babel/traverse").default;

function parseJS(path) {
  const src = fs.readFileSync(path, "utf8");
  const lines = src.split("\n");

  let ast;
  try {
    ast = parser.parse(src, {
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
  } catch (err) {
    return { file: { path, error: err.message }, imports: [], functions: [], classes: [], methods: [], code: src };
  }

  const imports = [];
  const functions = [];
  const classes = [];
  const methods = [];

  traverse(ast, {
    ImportDeclaration(pathNode) {
      imports.push(pathNode.node.source.value);
    },

    FunctionDeclaration(pathNode) {
      const n = pathNode.node;
      functions.push({
        name: n.id ? n.id.name : null,
        start_line: n.loc.start.line,
        end_line: n.loc.end.line,
        code: lines.slice(n.loc.start.line - 1, n.loc.end.line).join("\n"),
      });
    },

    ClassDeclaration(pathNode) {
      const n = pathNode.node;
      classes.push({
        name: n.id ? n.id.name : null,
        start_line: n.loc.start.line,
        end_line: n.loc.end.line,
        code: lines.slice(n.loc.start.line - 1, n.loc.end.line).join("\n"),
      });
    },

    ClassMethod(pathNode) {
      const n = pathNode.node;
      methods.push({
        name: n.key.name,
        start_line: n.loc.start.line,
        end_line: n.loc.end.line,
        code: lines.slice(n.loc.start.line - 1, n.loc.end.line).join("\n"),
      });
    },
  });

  return {
    file: { path },
    imports,
    functions,
    classes,
    methods,
    code: src,
  };
}

module.exports = { parseJS };
