"""
LLM Router: Direct queries to Ollama (local) or Groq (cloud) based on complexity
Includes: HyDE (Hypothetical Document Embedding), Query Rewriting, improved prompt building
"""

import os
import logging
from typing import Optional, Dict, List
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import ollama
except ImportError:
    ollama = None
    logger.warning("ollama package not installed")

try:
    from groq import Groq as GroqClient
except ImportError:
    GroqClient = None
    logger.warning("groq package not installed")


# ── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a document research assistant specializing in investigative and legal documents.

MANDATORY RULES:
1. CITATION REQUIRED: Every claim must cite its source.
   Format: [Source: <doc_id>, Page <n>, <source_type>]
2. NO SPECULATION: Only state what documents explicitly say.
3. OCR FLAGGING: If ocr_quality < 0.85, prefix your response with:
   [NOTE: Some sources may contain OCR transcription errors]
4. ALLEGATION FRAMING: Use 'Document X states/alleges...' not 'Person X did...'
5. REDACTION: If content is [REDACTED], explicitly mention it.
6. If context is insufficient, say so — do not fabricate or hallucinate.
7. Provide balanced, objective analysis citing multiple sources when available.
"""

# ── HyDE System Prompt ────────────────────────────────────────────────────────
# Used to generate a hypothetical answer document for better FAISS retrieval
HYDE_SYSTEM_PROMPT = """
You are a research assistant. Your task is to write a SHORT hypothetical document 
passage (2-4 sentences) that would PERFECTLY ANSWER the given question.

RULES:
- Write as if you are the actual document containing the answer
- Be specific and factual-sounding (even if invented)
- Match the style of investigative/legal documents
- Do NOT say "I think" or "possibly" — write as a factual document excerpt
- Keep it under 100 words
"""

# ── Query Rewriting System Prompt ────────────────────────────────────────────
QUERY_REWRITE_SYSTEM_PROMPT = """
You are a search query optimizer for a document retrieval system.
Given a user question, rewrite it into a cleaner, more specific search query 
that will retrieve better results from a vector database.

RULES:
- Remove filler words and conversational language
- Expand acronyms if known
- Add relevant synonyms or related terms in parentheses
- Keep it under 20 words
- Return ONLY the rewritten query, nothing else
"""


def build_prompt(query: str, retrieved_chunks: List[Dict]) -> str:
    """
    Build a structured prompt from query and retrieved chunks.
    Each chunk is numbered and labelled with its source for easy citation.

    Args:
        query:            the user's question
        retrieved_chunks: list of chunk dicts with 'text', 'source_file', 'page', 'doc_type'

    Returns:
        formatted prompt string ready to send to the LLM
    """
    context = ""

    for i, chunk in enumerate(retrieved_chunks, 1):
        source   = chunk.get("source_file", "Unknown")
        page     = chunk.get("page", 0)
        doc_type = chunk.get("doc_type", "text")
        text     = chunk.get("text", "")

        # Source label matching the citation format in SYSTEM_PROMPT
        context += f"\n--- Chunk {i} [Source: {source}, Page {page}, Type: {doc_type}] ---\n"
        context += text.strip() + "\n"

    if not context.strip():
        context = "[No matching documents found in the database.]"

    prompt = f"""CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Answer ONLY using the context above
- Cite every claim with [Source: doc_id, Page n, Type: type]
- If the context does not contain enough information, explicitly say so
- Do not fabricate or hallucinate any information
- Use 'Document states/alleges...' framing for sensitive claims

ANSWER:"""

    return prompt


class OllamaClient:
    """Wrapper around Ollama for local LLM inference"""

    def __init__(self, host: str = 'http://localhost:11434'):
        self.host      = host
        self.available = ollama is not None

        if not self.available:
            logger.warning("Ollama client not available (module not installed)")

    def generate(
        self,
        model: str = 'llama3:8b',
        prompt: str = '',
        system: str = SYSTEM_PROMPT,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Optional[str]:
        if not self.available:
            logger.error("Ollama not available")
            return None

        try:
            logger.info(f"Ollama: generating with {model}")
            response = ollama.chat(
                model=model,
                messages=[
                    {'role': 'system', 'content': system},
                    {'role': 'user',   'content': prompt}
                ],
                stream=False,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens
                }
            )
            return response['message']['content']

        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return None


class GroqClientWrapper:
    """Wrapper around Groq for cloud-based LLM inference.

    Supports multiple API keys via the GROQ_API_KEYS env var (comma-separated)
    or a single GROQ_API_KEY. When a key hits a rate limit, it is disabled for
    the lifetime of the process and the next key is tried automatically.
    """

    def __init__(self, api_key: Optional[str] = None):
        # Build list of keys in priority order:
        # 1) explicit api_key argument
        # 2) GROQ_API_KEYS (comma-separated)
        # 3) GROQ_API_KEY (single)
        self.client: Optional[GroqClient] = None
        self.available: bool = False

        if api_key:
            self.api_keys = [api_key]
        else:
            keys_env = os.getenv("GROQ_API_KEYS", "").strip()
            if keys_env:
                self.api_keys = [k.strip() for k in keys_env.split(",") if k.strip()]
            else:
                single = os.getenv("GROQ_API_KEY", "").strip()
                self.api_keys = [single] if single else []

        if not self.api_keys:
            logger.warning("No Groq API keys configured (GROQ_API_KEY(S)) - Groq client unavailable")
            return

        self._init_client(self.api_keys[0])

    def _init_client(self, api_key: str) -> None:
        try:
            self.client = GroqClient(api_key=api_key)
            self.available = True
            logger.info(
                "Groq client initialized with key ending ...%s",
                api_key[-4:] if len(api_key) >= 4 else "****",
            )
        except Exception as e:
            self.client = None
            self.available = False
            logger.error(f"Failed to initialize Groq client: {e}")

    def _rotate_key(self) -> bool:
        """
        Drop the current key and move to the next one.
        Returns True if a new key was activated, False if none remain.
        """
        if len(self.api_keys) <= 1:
            logger.warning("No backup Groq API keys left to rotate to")
            self.available = False
            self.client = None
            return False

        # "Delete" current key by removing it from the list
        finished_key = self.api_keys.pop(0)
        logger.warning(
            "Disabling exhausted Groq key ending ...%s and rotating to next",
            finished_key[-4:] if len(finished_key) >= 4 else "****",
        )

        self._init_client(self.api_keys[0])
        return self.available

    def generate(
        self,
        model: str = 'llama-3.3-70b-versatile',
        prompt: str = '',
        system: str = SYSTEM_PROMPT,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Optional[str]:
        if not self.available or self.client is None:
            logger.error("Groq client not available")
            return None

        try:
            logger.info(f"Groq: generating with {model}")
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': system},
                    {'role': 'user',   'content': prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content

        except Exception as e:
            msg = str(e)
            # If we clearly hit a rate limit, rotate to the next key (if any) and retry once
            if "rate_limit" in msg or "Rate limit reached" in msg or "rate_limit_exceeded" in msg:
                logger.error(f"Groq rate limit error: {e}")
                if self._rotate_key():
                    # Retry once with the new key
                    try:
                        logger.info("Retrying Groq call with rotated API key")
                        response = self.client.chat.completions.create(
                            model=model,
                            messages=[
                                {'role': 'system', 'content': system},
                                {'role': 'user',   'content': prompt}
                            ],
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        return response.choices[0].message.content
                    except Exception as e2:
                        logger.error(f"Groq error after key rotation: {e2}")
                        return None

            logger.error(f"Groq error: {e}")
            return None


class LLMRouter:
    """Route queries to appropriate LLM based on complexity"""

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        ollama_host: str = 'http://localhost:11434'
    ):
        self.ollama = OllamaClient(host=ollama_host)
        self.groq   = GroqClientWrapper(api_key=groq_api_key)

    # ── Internal LLM call (small tasks: HyDE, query rewriting) ───────────────
    def _quick_generate(self, system: str, user_prompt: str) -> Optional[str]:
        """
        Internal fast LLM call used for HyDE and query rewriting.
        Prefers Groq (faster), falls back to Ollama.
        Uses smaller max_tokens since output is always short.
        """
        if self.groq.available:
            return self.groq.generate(
                prompt=user_prompt,
                system=system,
                temperature=0.3,
                max_tokens=200,
            )
        elif self.ollama.available:
            return self.ollama.generate(
                prompt=user_prompt,
                system=system,
                temperature=0.3,
                max_tokens=200,
            )
        return None

    # ── Query Rewriting ────────────────────────────────────────────────────────
    def rewrite_query(self, query: str) -> str:
        """
        Rewrite the user query into a cleaner, more retrieval-friendly form.

        Example:
            Input:  "can you tell me what happened with the flights involving epstein?"
            Output: "Epstein flights passenger manifest aircraft records"

        Falls back to original query if rewriting fails.
        """
        logger.info(f"Rewriting query: {query[:60]}...")

        rewritten = self._quick_generate(
            system=QUERY_REWRITE_SYSTEM_PROMPT,
            user_prompt=f"Rewrite this query for document retrieval:\n\n{query}"
        )

        if rewritten and len(rewritten.strip()) > 3:
            rewritten = rewritten.strip().strip('"').strip("'")
            logger.info(f"  Original:  {query}")
            logger.info(f"  Rewritten: {rewritten}")
            return rewritten

        logger.warning("Query rewriting failed — using original query")
        return query

    # ── Multi-Query Expansion ──────────────────────────────────────────────────
    def generate_multi_queries(self, query: str, num_queries: int = 3) -> list:
        """
        Generate multiple semantically equivalent but differently-phrased query
        variations to improve recall during retrieval.

        Returns a deduplicated list of up to `num_queries` variants including the
        rewritten form of the original.  Falls back gracefully if LLM calls fail.
        """
        logger.info(f"Generating {num_queries} query variants for: {query[:60]}...")

        variations: list[str] = [query]

        # Rewritten form counts as a variant
        rewritten = self.rewrite_query(query)
        if rewritten and rewritten != query:
            variations.append(rewritten)

        # Generate additional variants until we reach num_queries
        attempts = 0
        while len(variations) < num_queries and attempts < num_queries:
            attempts += 1
            variant = self._quick_generate(
                system=(
                    "Generate a single search query variation. "
                    "Return ONLY the query, no explanation or punctuation."
                ),
                user_prompt=(
                    f"Create a semantically equivalent but differently-worded "
                    f"version of this search query:\n\n{query}"
                ),
            )
            if variant:
                variant = variant.strip().strip('"').strip("'")
                if len(variant) > 5 and variant not in variations:
                    variations.append(variant)

        logger.info(f"  Generated {len(variations)} query variants")
        return variations[:num_queries]

    # ── HyDE (Hypothetical Document Embedding) ─────────────────────────────────
    def generate_hyde_document(self, query: str) -> Optional[str]:
        """
        HyDE: Generate a hypothetical document passage that would answer the query.
        This hypothetical document is then embedded and used for FAISS retrieval
        INSTEAD OF embedding the raw query — improving semantic search quality.

        Why it works:
            - Raw queries are short and may not match document vocabulary
            - A hypothetical answer is in document-space vocabulary
            - Even if factually wrong, it captures the right semantic neighborhood

        Example:
            Query: "Who flew on Epstein's plane in 2002?"
            HyDE doc: "Flight records from 2002 show passenger manifests listing
                        several notable individuals aboard N909JE traveling between
                        Palm Beach and New York..."

        Falls back to None if generation fails (caller should use raw query embedding).
        """
        logger.info("Generating HyDE hypothetical document...")

        hyde_doc = self._quick_generate(
            system=HYDE_SYSTEM_PROMPT,
            user_prompt=f"Write a hypothetical document passage that answers:\n\n{query}"
        )

        if hyde_doc and len(hyde_doc.strip()) > 20:
            logger.info(f"  HyDE document: {hyde_doc[:100]}...")
            return hyde_doc.strip()

        logger.warning("HyDE generation failed — will use raw query for embedding")
        return None

    # ── Complexity Routing ────────────────────────────────────────────────────
    def estimate_complexity(self, context: str, query: str) -> Dict:
        context_tokens   = len(context.split()) // 1.33
        query_tokens     = len(query.split())
        is_image_query   = any(k in query.lower() for k in ['image', 'photo', 'picture', 'visual'])
        is_long_context  = context_tokens > 3000
        is_complex_query = query_tokens > 20 or any(
            k in query.lower() for k in ['compare', 'analyze', 'summarize', 'relationship']
        )
        return {
            'context_tokens':   context_tokens,
            'query_tokens':     query_tokens,
            'is_image_query':   is_image_query,
            'is_long_context':  is_long_context,
            'is_complex_query': is_complex_query
        }

    def route(self, context: str, query: str) -> str:
        """Decide which LLM to use. Returns 'ollama' or 'groq'."""
        complexity = self.estimate_complexity(context, query)

        if complexity['is_image_query']:
            return 'ollama'   # llava:13b for images
        if complexity['is_long_context']:
            return 'groq'     # better at long contexts
        if complexity['is_complex_query']:
            return 'groq'     # 70B model for complex reasoning

        return 'ollama'       # default to local for speed

    # ── Main Generate ─────────────────────────────────────────────────────────
    def generate(
        self,
        context: str,
        query: str,
        retrieved_chunks: Optional[List[Dict]] = None,
        force_provider: Optional[str] = None,
    ) -> Dict:
        """
        Generate answer with routing and structured prompt.

        Args:
            context:          raw context string (used for routing decisions)
            query:            user's original question
            retrieved_chunks: list of chunk dicts (preferred — produces better prompts)
                              if None, falls back to raw context string in prompt
            force_provider:   'groq', 'ollama', or 'auto'. If 'auto', triggers routing logic.

        Returns:
            {
                'answer':   str,
                'provider': 'ollama' or 'groq',
                'model':    str,
                'error':    Optional[str]
            }
        """
        try:
            # Provider selection:
            # - 'groq'   → always Groq
            # - 'ollama' → always Ollama
            # - 'auto'   → always Groq (as requested), falling back to routing
            #              only if Groq is unavailable
            # - None/other → legacy routing behaviour
            if force_provider == "groq":
                provider = "groq"
            elif force_provider == "ollama":
                provider = "ollama"
            elif force_provider == "auto":
                provider = "groq" if self.groq.available else self.route(context, query)
            else:
                provider = force_provider or self.route(context, query)

            # ── Build structured prompt ────────────────────────────────────
            if retrieved_chunks:
                # Preferred: use numbered chunks with source labels
                prompt = build_prompt(query, retrieved_chunks)
            else:
                # Fallback: raw context dump (old behaviour)
                prompt = (
                    f"CONTEXT:\n{context}\n\n"
                    f"QUESTION: {query}\n\n"
                    f"INSTRUCTIONS:\n"
                    f"- Answer ONLY using the context above\n"
                    f"- Cite sources using [Source: doc_id, Page n] format\n"
                    f"- If context is insufficient, explicitly say so\n"
                    f"- Do not fabricate any information\n\n"
                    f"ANSWER:"
                )

            # ── Generate ───────────────────────────────────────────────────
            if provider == 'groq':
                answer = self.groq.generate(
                    prompt=prompt,
                    system=SYSTEM_PROMPT,
                    temperature=0.1,
                    max_tokens=2048,
                )
                model = 'llama-3.3-70b-versatile'
            else:
                answer = self.ollama.generate(
                    prompt=prompt,
                    system=SYSTEM_PROMPT,
                    temperature=0.1,
                    max_tokens=2048,
                )
                model = 'llama3:8b'

            if answer is None:
                return {
                    'answer':   'Error: Failed to generate response',
                    'provider': provider,
                    'model':    model,
                    'error':    'Generation failed'
                }

            return {
                'answer':   answer,
                'provider': provider,
                'model':    model,
                'error':    None
            }

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {
                'answer':   f'Error: {str(e)}',
                'provider': 'unknown',
                'model':    'unknown',
                'error':    str(e)
            }


if __name__ == '__main__':
    router = LLMRouter()

    query = "Who was present at the meeting on January 15?"

    # Test query rewriting
    rewritten = router.rewrite_query(query)
    print(f"Rewritten query: {rewritten}")

    # Test HyDE
    hyde_doc = router.generate_hyde_document(query)
    print(f"HyDE document: {hyde_doc}")

    # Test full generation with mock chunks
    mock_chunks = [{
        "text": "The document states that John Smith was present at the meeting on January 15, 2024.",
        "source_file": "meeting_notes.pdf",
        "page": 3,
        "doc_type": "native"
    }]

    result = router.generate(
        context="",
        query=query,
        retrieved_chunks=mock_chunks
    )
    print(f"\nProvider: {result['provider']}")
    print(f"Model:    {result['model']}")
    print(f"Answer:   {result['answer']}")