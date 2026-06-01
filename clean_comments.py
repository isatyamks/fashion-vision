import os
import tokenize
import io

def remove_comments_and_docstrings(source):
    io_obj = io.StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0

    try:
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
                
            if token_type == tokenize.COMMENT:
                # Skip comments
                pass
            elif token_type == tokenize.STRING:
                # If it's a standalone string (not assigned), it's a docstring -> skip
                if prev_toktype != tokenize.INDENT and prev_toktype != tokenize.NEWLINE and start_col > 0:
                    out += token_string
            else:
                out += token_string
                
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
            
    except tokenize.TokenError:
        return source # Fallback if syntax error

    # Clean up excess blank lines left behind
    cleaned_lines = [line for line in out.splitlines() if line.strip() != ""]
    return "\n".join(cleaned_lines) + "\n"

def process_directory(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    source = f.read()
                
                cleaned = remove_comments_and_docstrings(source)
                
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(cleaned)
                count += 1
                print(f"Cleaned: {filepath}")
    return count

if __name__ == "__main__":
    src_dir = os.path.join(os.path.dirname(__file__), "src")
    print(f"Starting cleanup in {src_dir} ...")
    total = process_directory(src_dir)
    print(f"\nDone! Successfully removed comments and docstrings from {total} Python files.")
