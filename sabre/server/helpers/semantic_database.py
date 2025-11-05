"""
Enhanced database helpers with semantic understanding using OpenAI Vector Stores.

This module creates semantic representations of database schemas, table contents,
and relationships, enabling natural language queries and intelligent suggestions.
"""

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from openai import OpenAI


class SemanticDatabaseHelpers:
    """
    Enhanced database helpers with semantic understanding using OpenAI Vector Stores.

    This class creates semantic representations of database schemas, table contents,
    and relationships, enabling natural language queries and intelligent suggestions.
    """

    # Class-level cache for schemas, business context, and vector store IDs
    _schema_cache = {}
    _business_context_cache = {}
    _vector_store_cache = {}
    _cache_dir = None  # Will be set on first use

    @classmethod
    def _get_cache_dir(cls):
        """Get cache directory using SABRE's XDG paths"""
        if cls._cache_dir is None:
            from sabre.common.paths import get_data_dir

            cls._cache_dir = get_data_dir() / "semantic_database_cache"
        return cls._cache_dir

    @classmethod
    def _ensure_cache_dir(cls):
        """Ensure the cache directory exists"""
        cache_dir = cls._get_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _get_cache_key(cls, db_path: str) -> str:
        """Generate a cache key for a database path"""
        return f"{Path(db_path).resolve()}"

    @classmethod
    def _save_cache_to_disk(cls):
        """Save in-memory cache to disk for persistence"""
        cls._ensure_cache_dir()
        cache_dir = cls._get_cache_dir()

        # Save schema cache
        schema_file = cache_dir / "schemas.json"
        with open(schema_file, "w") as f:
            json.dump(cls._schema_cache, f, indent=2)

        # Save business context cache
        context_file = cache_dir / "business_context.json"
        with open(context_file, "w") as f:
            json.dump(cls._business_context_cache, f, indent=2)

        # Save vector store cache
        vector_store_file = cache_dir / "vector_stores.json"
        with open(vector_store_file, "w") as f:
            json.dump(cls._vector_store_cache, f, indent=2)

    @classmethod
    def _load_cache_from_disk(cls):
        """Load cache from disk into memory"""
        cls._ensure_cache_dir()
        cache_dir = cls._get_cache_dir()

        # Load schema cache
        schema_file = cache_dir / "schemas.json"
        if schema_file.exists():
            with open(schema_file, "r") as f:
                cls._schema_cache = json.load(f)

        # Load business context cache
        context_file = cache_dir / "business_context.json"
        if context_file.exists():
            with open(context_file, "r") as f:
                cls._business_context_cache = json.load(f)

        # Load vector store cache
        vector_store_file = cache_dir / "vector_stores.json"
        if vector_store_file.exists():
            with open(vector_store_file, "r") as f:
                cls._vector_store_cache = json.load(f)

    @classmethod
    def _get_openai_client(cls) -> OpenAI:
        """Get OpenAI client instance"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        return OpenAI(api_key=api_key)

    @staticmethod
    def create_semantic_understanding(db_path: str, force_refresh: bool = False) -> str:
        """
        Create semantic understanding of a database using OpenAI Vector Stores.

        This function:
        1. Analyzes database schema and relationships
        2. Extracts sample data and column meanings
        3. Creates rich semantic descriptions
        4. Uploads to OpenAI Vector Store for semantic search

        Example:
        result = SemanticDatabaseHelpers.create_semantic_understanding('./ecommerce.db')

        :param db_path: Path to the database file
        :param force_refresh: Force recreation of vector store even if cached
        :return: Status message with vector store information
        """
        # Load existing cache from disk
        SemanticDatabaseHelpers._load_cache_from_disk()
        cache_key = SemanticDatabaseHelpers._get_cache_key(db_path)

        # Check if we already have semantic understanding for this database
        if not force_refresh and cache_key in SemanticDatabaseHelpers._vector_store_cache:
            vector_store_id = SemanticDatabaseHelpers._vector_store_cache[cache_key]["vector_store_id"]
            return f"Database semantic understanding already exists for {db_path}. Vector store ID: {vector_store_id}"

        try:
            client = SemanticDatabaseHelpers._get_openai_client()
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Get database name for vector store
            db_name = Path(db_path).stem

            # Analyze database structure
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]

            # Create comprehensive semantic documentation
            semantic_docs = []

            # Database overview
            total_tables = len(tables)
            overview = f"""# Database: {db_name}

## Overview
This database contains {total_tables} tables and appears to be a {"business/ecommerce" if any(word in db_name.lower() for word in ["shop", "store", "ecommerce", "commerce"]) else "data"} database.

## Tables and Structure
"""
            semantic_docs.append(overview)

            # Analyze each table in detail
            for table in tables:
                # Get column information
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()

                # Get foreign key information
                cursor.execute(f"PRAGMA foreign_key_list({table})")
                foreign_keys = cursor.fetchall()

                # Get sample data (first 5 rows)
                cursor.execute(f"SELECT * FROM {table} LIMIT 5")
                sample_rows = cursor.fetchall()

                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]

                # Create semantic description for this table
                table_doc = f"""
### Table: {table}
**Purpose**: {SemanticDatabaseHelpers._infer_table_purpose(table, columns)}
**Row Count**: {row_count}

#### Columns:
"""
                for col in columns:
                    col_name, col_type, not_null, _, pk = (
                        col[1],
                        col[2],
                        col[3],
                        col[4],
                        col[5],
                    )
                    purpose = SemanticDatabaseHelpers._infer_column_purpose(col_name, col_type)
                    table_doc += f"- **{col_name}** ({col_type}): {purpose}"
                    if pk:
                        table_doc += " [PRIMARY KEY]"
                    if not_null:
                        table_doc += " [NOT NULL]"
                    table_doc += "\n"

                # Add relationships
                if foreign_keys:
                    table_doc += "\n#### Relationships:\n"
                    for fk in foreign_keys:
                        table_doc += f"- {fk[3]} references {fk[2]}.{fk[4]}\n"

                # Add sample data analysis
                if sample_rows:
                    table_doc += "\n#### Sample Data Patterns:\n"
                    table_doc += SemanticDatabaseHelpers._analyze_sample_data(columns, sample_rows)

                semantic_docs.append(table_doc)

            # Detect business domain and add domain-specific insights
            domain_analysis = SemanticDatabaseHelpers._analyze_business_domain(tables, semantic_docs)
            semantic_docs.append(f"\n## Business Domain Analysis\n{domain_analysis}")

            # Create a comprehensive semantic document
            full_semantic_doc = "\n".join(semantic_docs)

            # Create vector store and upload semantic understanding
            vector_store = client.vector_stores.create(
                name=f"Semantic DB: {db_name}",
                expires_after={
                    "anchor": "last_active_at",
                    "days": 30,
                },  # Auto-cleanup after 30 days
            )

            # Create temporary file with semantic documentation
            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as temp_file:
                temp_file.write(full_semantic_doc)
                temp_file_path = temp_file.name

            try:
                # Upload file to vector store
                with open(temp_file_path, "rb") as file_stream:
                    file_upload = client.files.create(file=file_stream, purpose="assistants")

                # Add file to vector store
                client.vector_stores.files.create(vector_store_id=vector_store.id, file_id=file_upload.id)

                # Cache the vector store information
                SemanticDatabaseHelpers._vector_store_cache[cache_key] = {
                    "vector_store_id": vector_store.id,
                    "created_at": datetime.now().isoformat(),
                    "file_id": file_upload.id,
                    "db_name": db_name,
                    "tables": tables,
                    "semantic_doc_length": len(full_semantic_doc),
                }

                # Also update traditional cache
                SemanticDatabaseHelpers._schema_cache[cache_key] = {
                    "discovered_at": datetime.now().isoformat(),
                    "tables": {table: {"row_count": 0} for table in tables},  # Simplified for this version
                    "vector_store_id": vector_store.id,
                }

                SemanticDatabaseHelpers._save_cache_to_disk()

                return f"""Semantic understanding created successfully for {db_path}

Vector Store ID: {vector_store.id}
Tables analyzed: {len(tables)}
Semantic document size: {len(full_semantic_doc)} characters
File ID: {file_upload.id}

The database semantic knowledge is now available for intelligent querying and analysis."""

            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)

            conn.close()

        except Exception as e:
            return f"Error creating semantic understanding for {db_path}: {str(e)}"

    @staticmethod
    def get_semantic_context(db_path: str, natural_language_query: str) -> str:
        """
        Search the semantic vector store for relevant database context.

        Example:
        context = SemanticDatabaseHelpers.get_semantic_context('./ecommerce.db',
                                                              'What are the top customers by revenue?')

        :param db_path: Path to the database file
        :param natural_language_query: Natural language question about the database
        :return: Relevant semantic context from the vector store
        """
        SemanticDatabaseHelpers._load_cache_from_disk()
        cache_key = SemanticDatabaseHelpers._get_cache_key(db_path)

        if cache_key not in SemanticDatabaseHelpers._vector_store_cache:
            return "No semantic understanding found for this database. Run create_semantic_understanding() first."

        try:
            client = SemanticDatabaseHelpers._get_openai_client()
            vector_store_id = SemanticDatabaseHelpers._vector_store_cache[cache_key]["vector_store_id"]

            # Search the vector store directly using the search method
            search_results = client.vector_stores.search(
                vector_store_id=vector_store_id,
                query=natural_language_query,
                max_num_results=5,  # Return top 5 results
            )

            # Extract and format the search results
            if not search_results or not hasattr(search_results, "data"):
                return "No relevant semantic context found for this query."

            context_chunks = []
            for i, result in enumerate(search_results.data, 1):
                # Format each result with score and content
                score = getattr(result, "score", 0.0) if hasattr(result, "score") else 0.0
                content = getattr(result, "content", "") if hasattr(result, "content") else str(result)

                context_chunks.append(f"### Result {i} (Score: {score:.3f})\n{content}")

            if not context_chunks:
                return "No relevant semantic context found for this query."

            context = "\n\n---\n\n".join(context_chunks)

            db_name = SemanticDatabaseHelpers._vector_store_cache[cache_key].get("db_name", "Unknown")

            return f"""Semantic database context for: '{natural_language_query}'
Database: {db_name}

{context}"""

        except Exception as e:
            return f"Error searching semantic context: {str(e)}"

    @staticmethod
    def generate_verification_queries(db_path: str, original_query: str) -> List[str]:
        """
        Generate alternative SQL queries that should yield consistent results for verification.

        Example:
        verification_queries = SemanticDatabaseHelpers.generate_verification_queries(
            './ecommerce.db', 'Top 10 customers by revenue'
        )

        :param db_path: Path to the database file
        :param original_query: The original natural language query to verify
        :return: List of alternative query approaches for cross-validation
        """
        context = SemanticDatabaseHelpers.get_semantic_context(
            db_path,
            f"Generate 3-5 alternative SQL approaches to answer: '{original_query}'. "
            f"Focus on different table joins, aggregation methods, or calculation approaches "
            f"that should yield the same or consistent results. Consider: "
            f"1. Different join paths to the same data "
            f"2. Alternative aggregation methods "
            f"3. Subquery vs CTE approaches "
            f"4. Different date/time groupings "
            f"5. Cross-validation through inverse queries",
        )

        if "Error searching semantic context" in context:
            return [f"Could not generate verification queries: {context}"]

        return [context]

    @staticmethod
    def get_semantic_suggestions(db_path: str, context: str = "general analysis") -> List[str]:
        """
        Get intelligent suggestions for database analysis based on semantic understanding.

        Example:
        suggestions = SemanticDatabaseHelpers.get_semantic_suggestions('./ecommerce.db', 'revenue analysis')

        :param db_path: Path to the database file
        :param context: Context for suggestions (e.g., 'revenue analysis', 'customer insights')
        :return: List of suggested analyses and queries
        """
        result = SemanticDatabaseHelpers.get_semantic_context(
            db_path,
            f"Suggest 5-10 meaningful SQL queries for {context}. Focus on business insights and actionable metrics.",
        )
        return [result]

    # Helper methods for semantic analysis
    @staticmethod
    def _infer_table_purpose(table_name: str, columns: List) -> str:
        """Infer the business purpose of a table based on name and columns"""
        name_lower = table_name.lower()
        col_names = [col[1].lower() for col in columns]

        if any(word in name_lower for word in ["user", "customer", "client"]):
            return "Stores information about users/customers"
        elif any(word in name_lower for word in ["order", "purchase", "transaction"]):
            return "Records sales transactions and orders"
        elif any(word in name_lower for word in ["product", "item", "inventory"]):
            return "Manages product/inventory information"
        elif any(word in name_lower for word in ["payment", "billing", "invoice"]):
            return "Handles payment and billing information"
        elif "id" in col_names and len(col_names) > 3:
            return f"Main entity table for {table_name} with detailed attributes"
        else:
            return f"Data table for {table_name} operations"

    @staticmethod
    def _infer_column_purpose(col_name: str, col_type: str) -> str:
        """Infer the purpose of a column based on name and type"""
        name_lower = col_name.lower()

        if name_lower.endswith("_id") or name_lower == "id":
            return "Unique identifier"
        elif "email" in name_lower:
            return "Email address"
        elif "name" in name_lower:
            return "Name/label field"
        elif "date" in name_lower or "time" in name_lower:
            return "Timestamp/date field"
        elif "price" in name_lower or "amount" in name_lower or "cost" in name_lower:
            return "Monetary value"
        elif "phone" in name_lower:
            return "Phone number"
        elif "address" in name_lower:
            return "Address information"
        elif "status" in name_lower:
            return "Status/state indicator"
        elif col_type.upper() in ["TEXT", "VARCHAR"]:
            return "Text/string data"
        elif col_type.upper() in ["INTEGER", "INT"]:
            return "Numeric value"
        elif col_type.upper() in ["REAL", "DECIMAL", "FLOAT"]:
            return "Decimal/floating point number"
        else:
            return f"{col_type} field"

    @staticmethod
    def _analyze_sample_data(columns: List, sample_rows: List) -> str:
        """Analyze sample data to provide insights"""
        if not sample_rows:
            return "No sample data available"

        analysis = f"Sample of {len(sample_rows)} rows:\n"
        col_names = [col[1] for col in columns]

        # Look for patterns in the data
        for i, col_name in enumerate(col_names):
            values = [str(row[i]) if row[i] is not None else "NULL" for row in sample_rows]
            unique_values = len(set(values))

            if unique_values == 1:
                analysis += f"  - {col_name}: Constant value ({values[0]})\n"
            elif unique_values == len(values):
                analysis += f"  - {col_name}: All unique values\n"
            else:
                analysis += f"  - {col_name}: {unique_values} unique values out of {len(values)}\n"

        return analysis

    @staticmethod
    def _analyze_business_domain(tables: List[str], semantic_docs: List[str]) -> str:
        """Analyze and determine the business domain of the database"""
        all_text = " ".join(tables + semantic_docs).lower()

        domains = {
            "ecommerce": [
                "order",
                "customer",
                "product",
                "cart",
                "payment",
                "shipping",
            ],
            "healthcare": ["patient", "doctor", "appointment", "medical", "treatment"],
            "education": ["student", "course", "grade", "enrollment", "teacher"],
            "finance": ["account", "transaction", "balance", "loan", "investment"],
            "hr": ["employee", "department", "salary", "performance", "attendance"],
        }

        domain_scores = {}
        for domain, keywords in domains.items():
            score = sum(1 for keyword in keywords if keyword in all_text)
            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            primary_domain = max(domain_scores, key=domain_scores.get)
            return f"Primary domain: {primary_domain.title()} (confidence: {domain_scores[primary_domain]}/{len(domains[primary_domain])} keywords matched)"
        else:
            return "Domain: General purpose database"

    @staticmethod
    def list_available_semantic_analyses() -> str:
        """
        List all available semantic database analyses stored in OpenAI vector stores.

        This function queries OpenAI's vector stores API and filters for stores
        containing semantic database analyses based on naming convention and metadata.

        Example:
        analyses = SemanticDatabaseHelpers.list_available_semantic_analyses()

        :return: Formatted list of available semantic analyses with metadata
        """
        try:
            client = SemanticDatabaseHelpers._get_openai_client()

            # Load local cache to cross-reference
            SemanticDatabaseHelpers._load_cache_from_disk()

            # List all vector stores
            vector_stores = client.vector_stores.list(limit=100)

            # Filter for semantic database stores (those with "Semantic DB:" prefix)
            semantic_stores = []
            for vs in vector_stores.data:
                # Check if this is a semantic database store by name pattern
                if vs.name and vs.name.startswith("Semantic DB:"):
                    # Extract database name from vector store name
                    db_name = vs.name.replace("Semantic DB:", "").strip()

                    # Get file count
                    file_count = vs.file_counts.total if hasattr(vs, "file_counts") else 0

                    # Get creation and last activity timestamps
                    created_at = (
                        datetime.fromtimestamp(vs.created_at).strftime("%Y-%m-%d %H:%M:%S")
                        if hasattr(vs, "created_at")
                        else "Unknown"
                    )
                    last_active = (
                        datetime.fromtimestamp(vs.last_active_at).strftime("%Y-%m-%d %H:%M:%S")
                        if hasattr(vs, "last_active_at")
                        else "Unknown"
                    )

                    # Check if we have local cache info
                    local_cache_info = None
                    for (
                        cache_key,
                        cache_data,
                    ) in SemanticDatabaseHelpers._vector_store_cache.items():
                        if cache_data.get("vector_store_id") == vs.id:
                            local_cache_info = cache_data
                            break

                    # Build store info
                    store_info = {
                        "vector_store_id": vs.id,
                        "db_name": db_name,
                        "created_at": created_at,
                        "last_active": last_active,
                        "file_count": file_count,
                        "status": vs.status if hasattr(vs, "status") else "unknown",
                        "local_db_path": cache_key if local_cache_info else None,
                        "tables": local_cache_info.get("tables", []) if local_cache_info else [],
                        "semantic_doc_length": local_cache_info.get("semantic_doc_length", 0)
                        if local_cache_info
                        else 0,
                    }

                    semantic_stores.append(store_info)

            if not semantic_stores:
                return "No semantic database analyses found in vector stores."

            # Format output
            output = f"Found {len(semantic_stores)} semantic database analysis(es):\n\n"

            for i, store in enumerate(semantic_stores, 1):
                output += f"{i}. {store['db_name']}\n"
                output += f"   Vector Store ID: {store['vector_store_id']}\n"
                output += f"   Created: {store['created_at']}\n"
                output += f"   Last Active: {store['last_active']}\n"
                output += f"   Status: {store['status']}\n"
                output += f"   Files: {store['file_count']}\n"

                if store["local_db_path"]:
                    output += f"   Local DB Path: {store['local_db_path']}\n"

                if store["tables"]:
                    table_list = ", ".join(store["tables"][:5])
                    if len(store["tables"]) > 5:
                        table_list += f", ... ({len(store['tables'])} total)"
                    output += f"   Tables: {table_list}\n"

                if store["semantic_doc_length"]:
                    output += f"   Semantic Doc Size: {store['semantic_doc_length']:,} characters\n"

                output += "\n"

            output += "\nTo use a semantic analysis, reference the database by name or provide the vector store ID to get_semantic_context()."

            return output

        except Exception as e:
            return f"Error listing semantic analyses: {str(e)}"

    @staticmethod
    def clear_semantic_cache(db_path: Optional[str] = None) -> str:
        """
        Clear semantic understanding cache for a specific database or all databases.

        Example:
        SemanticDatabaseHelpers.clear_semantic_cache('./ecommerce.db')  # Clear specific database
        SemanticDatabaseHelpers.clear_semantic_cache()  # Clear all cached data

        :param db_path: Optional path to specific database, or None to clear all
        :return: Confirmation message
        """
        if db_path:
            cache_key = SemanticDatabaseHelpers._get_cache_key(db_path)
            SemanticDatabaseHelpers._schema_cache.pop(cache_key, None)
            SemanticDatabaseHelpers._business_context_cache.pop(cache_key, None)
            SemanticDatabaseHelpers._vector_store_cache.pop(cache_key, None)
            message = f"Cleared semantic cache for {db_path}"
        else:
            SemanticDatabaseHelpers._schema_cache.clear()
            SemanticDatabaseHelpers._business_context_cache.clear()
            SemanticDatabaseHelpers._vector_store_cache.clear()
            message = "Cleared all semantic database cache"

        SemanticDatabaseHelpers._save_cache_to_disk()
        return message
