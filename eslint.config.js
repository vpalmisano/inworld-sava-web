import js from '@eslint/js';
import { FlatCompat } from '@eslint/eslintrc';
import stylistic from '@stylistic/eslint-plugin';
import prettier from 'eslint-config-prettier';
import globals from 'globals';
import { fileURLToPath } from 'node:url';
import tseslint from 'typescript-eslint';

const compat = new FlatCompat({ baseDirectory: fileURLToPath(new URL('.', import.meta.url)) });

export default [
  { ignores: ['dist/**', 'node_modules/**', '.venv/**', 'export/.upstream/**', '_site/**'] },
  ...compat.extends('defaults'),
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ['**/*.{js,mjs,ts}'],
    languageOptions: {
      globals: { ...globals.browser, ...globals.worker, ...globals.node },
    },
  },
  prettier,
  {
    files: ['**/*.{js,mjs,ts}'],
    plugins: { '@stylistic': stylistic },
    rules: {
      '@stylistic/padding-line-between-statements': [
        'error',
        {
          blankLine: 'always',
          prev: [
            'function',
            'const',
            'let',
            'var',
            { selector: 'ExportNamedDeclaration[declaration.type="FunctionDeclaration"]' },
            { selector: 'ExportDefaultDeclaration[declaration.type="FunctionDeclaration"]' },
            { selector: 'ExportNamedDeclaration[declaration.type="VariableDeclaration"]' },
          ],
          next: '*',
        },
      ],
      '@stylistic/lines-between-class-members': [
        'error',
        { enforce: [{ blankLine: 'always', prev: 'method', next: '*' }] },
      ],
    },
  },
];
