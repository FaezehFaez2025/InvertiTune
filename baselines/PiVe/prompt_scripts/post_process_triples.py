#!/usr/bin/env python3
import os
import ast
import json
import argparse
import re

def get_readable_error(error, line_content):
    """
    Convert ast error to human readable format.
    
    Args:
        error: The exception object
        line_content (str): The original line content
        
    Returns:
        str: Human readable error message
    """
    error_str = str(error)
    
    if "malformed node or string" in error_str:
        # Try to identify the problematic part
        if "true" in line_content.lower():
            return "Contains JavaScript 'true' instead of Python 'True'"
        elif "false" in line_content.lower():
            return "Contains JavaScript 'false' instead of Python 'False'"
        elif "none" in line_content.lower():
            return "Contains 'None' or 'none' that needs to be quoted"
        else:
            return "Contains unquoted Python keywords or variables"
    elif "unexpected EOF" in error_str:
        return "Incomplete or malformed list structure"
    elif "invalid syntax" in error_str:
        return "Invalid syntax in triple structure"
    else:
        return f"Parse error: {error_str}"

def fix_line_content(line_content):
    """
    Fix common issues in line content.
    
    Args:
        line_content (str): Original line content
        
    Returns:
        str: Fixed line content
    """
    # Fix JavaScript booleans
    fixed = re.sub(r'\btrue\b', 'True', line_content)
    fixed = re.sub(r'\bfalse\b', 'False', fixed)
    
    # Fix unquoted None
    fixed = re.sub(r'\bNone\b', '"None"', fixed)
    fixed = re.sub(r'\bnone\b', '"none"', fixed)
    
    return fixed

def validate_and_fix_triple_format(input_file, output_file):
    """
    Validate and fix triple format in the input file.
    
    Args:
        input_file (str): Path to input file
        output_file (str): Path to output file
    """
    errors = []
    fixed_lines = []
    total_lines = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            total_lines += 1
            
            # Handle empty lines
            if not line:
                # Use a placeholder triple instead of an empty list
                fixed_lines.append('[["", "", ""]]')
                continue
                
            line_errors = []
            fixed_line = line
            
            try:
                # First try to parse the original line
                data = ast.literal_eval(line)
            except Exception as e:
                # If parsing fails, try to fix the line
                readable_error = get_readable_error(e, line)
                line_errors.append(f"Line {line_num}: {readable_error}")
                
                # Try to fix the line
                fixed_line = fix_line_content(line)
                
                try:
                    data = ast.literal_eval(fixed_line)
                    line_errors.append(f"Line {line_num}: Fixed successfully")
                except Exception as e2:
                    line_errors.append(f"Line {line_num}: Could not fix - {get_readable_error(e2, fixed_line)}")
                    # Use placeholder triple as fallback instead of skipping
                    data = [["", "", ""]]
                    line_errors.append(f"Line {line_num}: Using placeholder triple as fallback")
            
            # Validate the parsed data
            if not isinstance(data, list):
                line_errors.append(f"Line {line_num}: Expected list, got {type(data).__name__}")
                data = [["", "", ""]]  # Convert to placeholder triple
            
            # Fix any non-string elements in triples
            fixed_triples = []
            for triple_idx, triple in enumerate(data):
                if not isinstance(triple, list) or len(triple) != 3:
                    line_errors.append(f"Line {line_num}, triple {triple_idx}: Expected list of 3 elements, got {triple}")
                    continue
                
                # Convert all elements to strings
                fixed_triple = []
                for elem_idx, element in enumerate(triple):
                    if element is None:
                        fixed_element = ""
                    elif isinstance(element, (int, float, bool)):
                        fixed_element = str(element)
                    elif isinstance(element, str):
                        fixed_element = element
                    else:
                        fixed_element = str(element)
                    
                    fixed_triple.append(fixed_element)
                
                fixed_triples.append(fixed_triple)
            
            # If no valid triples remain, use a placeholder triple instead of an empty list
            if not fixed_triples:
                fixed_triples = [["", "", ""]]

            # Create the final fixed line (always create a line)
            final_line = json.dumps(fixed_triples)
            fixed_lines.append(final_line)
            
            if line_errors:
                errors.extend(line_errors)
    
    # Write the fixed file - ensure same number of lines as input
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in fixed_lines:
            f.write(line + '\n')
    
    return errors, total_lines

def main():
    parser = argparse.ArgumentParser(description="Post-process triple format in files")
    parser.add_argument("--input_file", required=True, help="Input file path")
    parser.add_argument("--output_file", help="Output file path (default: input_file with _post_processed suffix)")
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if not args.output_file:
        base_name = os.path.splitext(args.input_file)[0]
        args.output_file = f"{base_name}_post_processed.txt"
    
    print(f"Processing file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    
    errors, total_lines = validate_and_fix_triple_format(args.input_file, args.output_file)
    
    print(f"\nProcessing complete!")
    print(f"Total lines processed: {total_lines}")
    print(f"Total errors found: {len(errors)}")
    
    if errors:
        print(f"\nErrors detected and fixed:")
        for error in errors:
            print(f"  - {error}")
    else:
        print(f"\nNo errors detected! All lines have correct format.")
    
    print(f"\nFixed file saved to: {args.output_file}")

if __name__ == "__main__":
    main() 