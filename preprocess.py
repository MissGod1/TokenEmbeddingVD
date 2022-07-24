#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   preprocess.py
@Time    :   2022/07/23 15:45:14
@Author  :   Baize
@Version :   1.0
@Contact :   
@License :   
@Desc    :   
'''

import warnings
warnings.filterwarnings('ignore')
import clang.cindex
import clang.enumerations

clang.cindex.Config.set_library_path("/usr/lib/x86_64-linux-gnu")
clang.cindex.Config.set_library_file('/usr/lib/x86_64-linux-gnu/libclang-6.0.so.1')

import re 
import os

keywords = ["alignas", "alignof", "and", "and_eq", "asm", "atomic_cancel", "atomic_commit", 
            "atomic_noexcept", "auto", "bitand", "bitor", "bool", "break", "case", "catch", 
            "char", "char8_t", "char16_t", "char32_t", "class", "compl", "concept", "const", 
            "consteval", "constexpr", "constinit", "const_cast", "continue", "co_await", 
            "co_return", "co_yield", "decltype", "default", "delete", "do", "double", "dynamic_cast", 
            "else", "enum", "explicit", "export", "extern", "false", "float", "for", "friend", "goto", 
            "if", "inline", "int", "long", "mutable", "namespace", "new", "noexcept", "not", "not_eq", 
            "nullptr", "operator", "or", "or_eq", "private", "protected", "public", "reflexpr", 
            "register", "reinterpret_cast", "requires", "return", "short", "signed", "sizeof", "static", 
            "static_assert", "static_cast", "struct", "switch", "synchronized", "template", "this", 
            "thread_local", "throw", "true", "try", "typedef", "typeid", "typename", "union", "unsigned", 
            "using", "virtual", "void", "volatile", "wchar_t", "while", "xor", "xor_eq", "NULL"]

class Tokenizer:
    # creates the object, does the inital parse
    def __init__(self, path, tokenizer_type='original'):
        self.index = clang.cindex.Index.create()
        self.tu = self.index.parse(path)
        self.path = self.extract_path(path)
        self.symbol_table = {}
        self.symbol_count = 1
        self.tokenizer_type = tokenizer_type

    # To output for split_functions, must have same path up to last two folders
    def extract_path(self, path):
        return "".join(path.split("/")[:-2])

    
    def full_tokenize_cursor(self, cursor):
        tokens = cursor.get_tokens()
        result = []
        for token in tokens:
            if token.kind.name == "COMMENT":
                continue
            if token.kind.name == "LITERAL":
                result += self.process_literal(token)
                continue
            if token.kind.name == "IDENTIFIER":
                if token.spelling in keywords:
                    result += [token.spelling]
                else:
                    result += ["ID"]
                continue
            result += [token.spelling]
        return result

    def full_tokenize(self):
        cursor = self.tu.cursor
        return self.full_tokenize_cursor(cursor)

    def process_literal(self, literal):
        cursor_kind = clang.cindex.CursorKind
        kind = literal.cursor.kind
        if kind == cursor_kind.INTEGER_LITERAL:
            return literal.spelling
        if kind == cursor_kind.FLOATING_LITERAL:
            return literal.spelling
        if kind == cursor_kind.IMAGINARY_LITERAL:
            return ["NUM"]       
        if kind == cursor_kind.STRING_LITERAL:
            return ["STRING"]
        sp = literal.spelling
        if re.match('[0-9]+', sp) is not None:
            return sp
        return ["LITERAL"]

    def split_functions(self, method_only):
        # results = []
        cursor_kind = clang.cindex.CursorKind
        cursor = self.tu.cursor
        for c in cursor.get_children():
            filename = c.location.file.name if c.location.file != None else "NONE"
            extracted_path = self.extract_path(filename)

            if (c.kind == cursor_kind.CXX_METHOD or (method_only == False and c.kind == cursor_kind.FUNCTION_DECL)) and extracted_path == self.path:
                name = c.spelling
                tokens = self.full_tokenize_cursor(c)
                filename = filename.split("/")[-1]
                return tokens
                # results += [tokens]
                # break;
        # return results
        return None
    

def tokenize(file_text, filename):
    try:
        c_file = open(f'/tmp/{filename}.c', 'w')
        c_file.write(file_text)
        c_file.close()
        tok = Tokenizer(f'/tmp/{filename}.c')
        results = tok.split_functions(False)
        # print(results)
        # return ' '.join(results[0])
        os.remove(f'/tmp/{filename}.c')
        return results
    except:
        return None

if __name__ == '__main__':
    print(tokenize("int main(){\n\tint *a = new int[10];\n\treturn 50;\n}\n", 'test1.c'))