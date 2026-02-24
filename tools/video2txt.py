#!/usr/bin/env python3
"""
This module converts video files to text using OpenAI's Whisper model for transcription
and formats the output using an LLM (OpenAI/DeepSeek API).

Requirements:
- openai-whisper
- openai
"""

import os
import argparse
import whisper

from dotenv import load_dotenv
load_dotenv()

def load_env(*args, default=None):
    """Load environment variables."""
    for arg in args:
        if arg in os.environ:
            return os.environ.get(arg)
    return default

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Video to text converter")
    parser.add_argument("-o", "--output", type=str, help="Output text file path")
    parser.add_argument("-m", "--model", type=str, default="large", help="Model to use")
    parser.add_argument("input_file", type=str, help="Input video file")
    return parser.parse_args()

def load_format_tool():
    """Load format tool."""
    from openai import OpenAI
    default_prompt = """请将以下内容进行格式化，要求：
1.去除多余的空格和换行；
2.保持原有的语义不变；
3.如果内容中包含时间戳，请将其转换为标准的时间格式（例如：00:01:30）；
4.基于文本语义逻辑自动分段。
"""
    base_url = load_env('OPENAI_BASE_URL', default="https://api.deepseek.com")
    api_key = load_env('OPENAI_API_KEY', 'DEEPSEEK_API_KEY', default="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    model = load_env('OPENAI_MODEL', 'DEEPSEEK_MODEL', default="deepseek-chat")
    prompt = load_env('OPENAI_PROMPT', default=default_prompt)

    client = OpenAI(api_key=api_key, base_url=base_url)
    def format_content(content):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "user", "content": content}
            ]
        )
        return resp.choices[0].message.content
    return format_content

def main():
    """Main entry point of the program."""
    args = parse_arguments()
    whisper_model = whisper.load_model(args.model)
    content_formatter = load_format_tool()

    result = whisper_model.transcribe(args.input_file)
    content = content_formatter(result['text'])

    if args.output:
        with open(args.output, "w") as f:
            f.write(content)
    else:
        print(content)

if __name__ == "__main__":
    main()