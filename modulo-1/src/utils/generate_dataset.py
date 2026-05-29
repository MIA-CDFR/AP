"""
Script to rephrase descriptions from ag_news.csv using multiple LLM model families.
Supported providers: OpenAI, Meta (via Groq), Anthropic, Google
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import time

# Import client libraries
try:
    from openai import OpenAI
except ImportError:
    print("OpenAI not installed. Install with: pip install openai")

try:
    import anthropic
except ImportError:
    print("Anthropic not installed. Install with: pip install anthropic")

try:
    from groq import Groq
except ImportError:
    print("Groq not installed. Install with: pip install groq")

try:
    import google.genai
    import google.genai.types as types
except ImportError:
    print("Google Generative AI not installed. Install with: pip install google-genai")


class DescriptionRephrase:
    def __init__(self, openai_key: Optional[str] = None, anthropic_key: Optional[str] = None,
                 groq_key: Optional[str] = None, google_key: Optional[str] = None):
        """
        Initialize rephrase engine with API keys.
        
        Args:
            openai_key: OpenAI API key (or use OPENAI_API_KEY env var)
            anthropic_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
            groq_key: Groq API key for Meta models (or use GROQ_API_KEY env var)
            google_key: Google API key (or use GOOGLE_API_KEY env var)
        """
        # OpenAI
        self.openai_client = None
        openai_key = openai_key or os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.openai_client = OpenAI(api_key=openai_key)
        
        # Anthropic
        self.anthropic_client = None
        anthropic_key = anthropic_key or os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
        
        # Groq (Meta models)
        self.groq_client = None
        groq_key = groq_key or os.getenv("GROQ_API_KEY")
        if groq_key:
            self.groq_client = Groq(api_key=groq_key)
        
        # Google
        self.google_client = None
        google_key = google_key or os.getenv("GOOGLE_API_KEY")
        if google_key:
            self.google_client = google.genai.Client(api_key=google_key)
    
    @staticmethod
    def _build_batch_prompt(texts: List[str]) -> str:
        return (
            "Rephrase each text clearly and concisely while preserving meaning. "
            "Return ONLY a valid JSON array of strings with the same length and order as the input. "
            "If an input text is empty, return an empty string in that position.\n\n"
            f"Input JSON array:\n{json.dumps(texts, ensure_ascii=False)}"
        )

    @staticmethod
    def _parse_batch_response(raw_text: str, expected_len: int) -> List[Optional[str]]:
        try:
            parsed = json.loads(raw_text.replace("```json\n", "").replace("```", "").strip())
            if isinstance(parsed, list):
                normalized = [str(item).strip() if item is not None else "" for item in parsed]
                if len(normalized) < expected_len:
                    normalized.extend([""] * (expected_len - len(normalized)))
                return normalized[:expected_len]
        except Exception:
            pass

        return [None] * expected_len

    def rephrase_openai_batch(self, texts: List[str]) -> List[Optional[str]]:
        """Rephrase a batch using OpenAI GPT model."""
        if not self.openai_client:
            return [None] * len(texts)
        
        try:
            prompt = self._build_batch_prompt(texts)
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo", # openai/gpt-oss-120b
                messages=[
                    {"role": "system", "content": "You are an expert writer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=3000
            )
            raw_text = response.choices[0].message.content or ""
            return self._parse_batch_response(raw_text, len(texts))
        except Exception as e:
            print(f"OpenAI error: {e}")
            return [None] * len(texts)
    
    def rephrase_openai_groq_batch(self, texts: List[str]) -> List[Optional[str]]:
        """Rephrase a batch using OpenAI GPT model via Groq."""
        if not self.groq_client:
            return [None] * len(texts)

        try:
            prompt = self._build_batch_prompt(texts)
            response = self.groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": "You are an expert writer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=3000
            )
            raw_text = response.choices[0].message.content or ""
            return self._parse_batch_response(raw_text, len(texts))
        except Exception as e:
            print(f"OpenAI (Groq) error: {e}")
            return [None] * len(texts)

    def rephrase_anthropic_batch(self, texts: List[str]) -> List[Optional[str]]:
        """Rephrase a batch using Anthropic Claude model."""
        if not self.anthropic_client:
            return [None] * len(texts)
        
        try:
            prompt = self._build_batch_prompt(texts)
            response = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=3000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            raw_text = response.content[0].text if response.content else ""
            return self._parse_batch_response(raw_text, len(texts))
        except Exception as e:
            print(f"Anthropic error: {e}")
            return [None] * len(texts)
    
    def rephrase_meta_batch(self, texts: List[str]) -> List[Optional[str]]:
        """Rephrase a batch using Meta Llama model via Groq."""
        if not self.groq_client:
            return [None] * len(texts)
        
        try:
            prompt = self._build_batch_prompt(texts)
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are an expert writer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=3000
            )
            raw_text = response.choices[0].message.content or ""
            return self._parse_batch_response(raw_text, len(texts))
        except Exception as e:
            print(f"Meta (Groq) error: {e}")
            return [None] * len(texts)
    
    def rephrase_google_batch(self, texts: List[str]) -> List[Optional[str]]:
        """Rephrase a batch using Google Gemini model."""
        if not self.google_client:
            return [None] * len(texts)
        
        try:
            prompt = self._build_batch_prompt(texts)
            model = "gemini-flash-latest"
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]
            generate_content_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                     thinking_budget=0,
                ),
            )
            response = self.google_client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_content_config,
            )
            raw_text = response.text or ""
            return self._parse_batch_response(raw_text, len(texts))
        except Exception as e:
            print(f"Google error: {e}")
            return [None] * len(texts)

    def rephrase_all_batch(self, texts: List[str], delay: float = 0.0) -> Dict[str, List[Optional[str]]]:
        """Rephrase a batch using all available models."""
        results = {
            # "openai": self.rephrase_openai_batch(texts),
            "openai": self.rephrase_openai_groq_batch(texts),
            "anthropic": self.rephrase_anthropic_batch(texts),
            "meta": self.rephrase_meta_batch(texts),
            "google": self.rephrase_google_batch(texts),
        }
        if delay > 0:
            time.sleep(delay)
        return results


def process_csv(
    input_file: str,
    output_file: str,
    providers: List[str] = None,
    delay: float = 1.0,
    chunk_size: int = 25,
):
    """
    Process CSV file and rephrase descriptions.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        providers: List of providers to use (openai, anthropic, meta, google)
        delay: Delay between API calls in seconds to avoid rate limiting
        chunk_size: Number of rows to process per chunk
    """
    if providers is None:
        providers = ["openai", "anthropic", "meta", "google"]
    
    rephrase = DescriptionRephrase()
    
    # Read input file
    rows = []
    fieldnames = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    print(f"Loaded {len(rows)} rows from {input_file}")
    
    # Add new columns for rephrased descriptions
    new_fieldnames = fieldnames.copy()
    for provider in providers:
        new_fieldnames.append(f"description_rephrased_{provider}")
    
    # Process rows in chunks
    total_rows = len(rows)
    total_chunks = (total_rows + chunk_size - 1) // chunk_size

    for chunk_idx, chunk_start in enumerate(range(0, total_rows, chunk_size), start=1):
        chunk_end = min(chunk_start + chunk_size, total_rows)
        print(
            f"Processing chunk {chunk_idx}/{total_chunks} "
            f"(rows {chunk_start + 1}-{chunk_end})"
        )

        chunk_rows = rows[chunk_start:chunk_end]
        chunk_descriptions = [row.get("description", "") for row in chunk_rows]

        print(f"  Sending {len(chunk_rows)} descriptions per provider call...")
        chunk_results = rephrase.rephrase_all_batch(chunk_descriptions, delay=delay)

        for offset, row in enumerate(chunk_rows):
            row_idx = chunk_start + offset
            print(f"  Mapping row {row_idx + 1}/{total_rows}: {row.get('title', 'N/A')[:50]}...")

            for provider in providers:
                provider_values = chunk_results.get(provider, [])
                value = provider_values[offset] if offset < len(provider_values) else ""
                row[f"description_rephrased_{provider}"] = value or ""
    
    # Write output file
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Rephrased data saved to {output_file}")


def main():
    """Main entry point."""
    # Configuration
    input_file = "data/ag_news_small.csv"
    output_file = "data/ag_news_rephrased.csv"
    providers = ["openai", "anthropic", "meta", "google"]
    chunk_size = 25
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file {input_file} not found")
        return
    
    print("Starting description rephrase process...")
    print(f"Providers: {', '.join(providers)}")
    print()
    
    process_csv(input_file, output_file, providers, delay=2.0, chunk_size=chunk_size)
    
    print("\nProcess completed!")


if __name__ == "__main__":
    main()
