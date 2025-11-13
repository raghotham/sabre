"""
Database helpers for SABRE that provide intelligent SQL query capabilities
with persistent learning and caching of database schemas and business context.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from datetime import datetime


class DatabaseHelpers:
    """
    Database helpers for SABRE that provide intelligent SQL query capabilities
    with persistent learning and caching of database schemas and business context.
    """

    # Class-level cache for schemas and business context
    _schema_cache = {}
    _business_context_cache = {}
    _cache_dir = None  # Will be set on first use

    @classmethod
    def _get_cache_dir(cls):
        """Get cache directory using SABRE's XDG paths"""
        if cls._cache_dir is None:
            from sabre.common.paths import get_data_dir

            cls._cache_dir = get_data_dir() / "database_cache"
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

    @staticmethod
    def connect_database(db_path: str) -> str:
        """
        Connect to a database and perform initial discovery if not cached.
        This function learns about the database structure and caches it for future use.

        Example:
        connection_info = DatabaseHelpers.connect_database('./sample_ecommerce.db')

        :param db_path: Path to the database file
        :return: Connection status and summary information
        """
        # Load existing cache from disk
        DatabaseHelpers._load_cache_from_disk()

        cache_key = DatabaseHelpers._get_cache_key(db_path)

        # Check if we already know about this database
        if cache_key in DatabaseHelpers._schema_cache:
            return (
                f"Connected to database: {db_path}\n"
                + f"Database already discovered. Found {len(DatabaseHelpers._schema_cache[cache_key]['tables'])} tables."
            )

        # New database - perform discovery
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]

            schema_info = {
                "discovered_at": datetime.now().isoformat(),
                "tables": {},
                "relationships": [],
                "sample_data": {},
            }

            # Analyze each table
            for table in tables:
                # Get column information
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()

                # Get foreign key information
                cursor.execute(f"PRAGMA foreign_key_list({table})")
                foreign_keys = cursor.fetchall()

                # Get sample data (first 3 rows)
                cursor.execute(f"SELECT * FROM {table} LIMIT 3")
                sample_rows = cursor.fetchall()

                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]

                schema_info["tables"][table] = {
                    "columns": [
                        {
                            "name": col[1],
                            "type": col[2],
                            "notnull": col[3],
                            "pk": col[5],
                        }
                        for col in columns
                    ],
                    "foreign_keys": [
                        {
                            "column": fk[3],
                            "references_table": fk[2],
                            "references_column": fk[4],
                        }
                        for fk in foreign_keys
                    ],
                    "row_count": row_count,
                    "column_names": [col[1] for col in columns],
                }

                schema_info["sample_data"][table] = sample_rows

                # Store relationships
                for fk in foreign_keys:
                    schema_info["relationships"].append(
                        {
                            "from_table": table,
                            "from_column": fk[3],
                            "to_table": fk[2],
                            "to_column": fk[4],
                        }
                    )

            # Cache the discovered schema
            DatabaseHelpers._schema_cache[cache_key] = schema_info

            # Initialize business context
            DatabaseHelpers._business_context_cache[cache_key] = {
                "discovered_at": datetime.now().isoformat(),
                "business_domain": "unknown",
                "common_queries": [],
                "metrics": [],
                "insights": [],
            }

            # Save to disk
            DatabaseHelpers._save_cache_to_disk()

            conn.close()

            total_rows = sum(info["row_count"] for info in schema_info["tables"].values())
            summary_lines = [
                f"Connected and discovered database: {db_path}",
                f"Found {len(tables)} tables: {', '.join(tables) if tables else 'None'}",
                f"Total rows across all tables: {total_rows}",
            ]

            if schema_info.get("tables"):
                summary_lines.append("Columns by table:")
                for table_name, table_info in schema_info["tables"].items():
                    columns = ", ".join(col["name"] for col in table_info["columns"])
                    summary_lines.append(f"- {table_name}: {columns}")

            return "\n".join(summary_lines)

        except Exception as e:
            return f"Error connecting to database {db_path}: {str(e)}"

    @staticmethod
    def query_database(db_path: str, query: str, learn_from_query: bool = True) -> Union[pd.DataFrame, str]:
        """
        Execute a SQL query against the database and optionally learn from it.

        Example:
        df = DatabaseHelpers.query_database('./sample_ecommerce.db', 'SELECT * FROM customers LIMIT 5')

        :param db_path: Path to the database file
        :param query: SQL query to execute
        :param learn_from_query: Whether to store this query pattern for learning
        :return: Query results as a pandas DataFrame or error message
        """
        try:
            # Load cache
            DatabaseHelpers._load_cache_from_disk()
            cache_key = DatabaseHelpers._get_cache_key(db_path)

            # Execute query
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(query, conn)
            conn.close()

            # Learn from the query if requested
            if learn_from_query and cache_key in DatabaseHelpers._business_context_cache:
                query_lower = query.lower().strip()
                context = DatabaseHelpers._business_context_cache[cache_key]

                # Store common query patterns
                if query_lower not in [q.lower() for q in context["common_queries"]]:
                    context["common_queries"].append(query)

                    # Limit to last 50 queries
                    if len(context["common_queries"]) > 50:
                        context["common_queries"] = context["common_queries"][-50:]

                # Auto-detect business domain based on table names and queries
                if context["business_domain"] == "unknown":
                    if any(word in query_lower for word in ["customer", "order", "product", "sale", "purchase"]):
                        context["business_domain"] = "ecommerce"
                    elif any(word in query_lower for word in ["patient", "doctor", "medical", "hospital"]):
                        context["business_domain"] = "healthcare"
                    elif any(word in query_lower for word in ["student", "course", "grade", "enrollment"]):
                        context["business_domain"] = "education"

                DatabaseHelpers._save_cache_to_disk()

            return df

        except Exception as e:
            return f"Error executing query: {str(e)}"

    @staticmethod
    def get_database_schema(db_path: str) -> Dict[str, Any]:
        """
        Get the cached schema information for a database.

        Example:
        schema = DatabaseHelpers.get_database_schema('./sample_ecommerce.db')

        :param db_path: Path to the database file
        :return: Cached schema information
        """
        DatabaseHelpers._load_cache_from_disk()
        cache_key = DatabaseHelpers._get_cache_key(db_path)

        if cache_key not in DatabaseHelpers._schema_cache:
            # Auto-connect if not cached
            DatabaseHelpers.connect_database(db_path)
            DatabaseHelpers._load_cache_from_disk()

        return DatabaseHelpers._schema_cache.get(cache_key, {})

    @staticmethod
    def get_business_context(db_path: str) -> Dict[str, Any]:
        """
        Get the learned business context for a database.

        Example:
        context = DatabaseHelpers.get_business_context('./sample_ecommerce.db')

        :param db_path: Path to the database file
        :return: Business context and learned patterns
        """
        DatabaseHelpers._load_cache_from_disk()
        cache_key = DatabaseHelpers._get_cache_key(db_path)

        return DatabaseHelpers._business_context_cache.get(cache_key, {})

    @staticmethod
    def suggest_analysis_queries(db_path: str) -> List[str]:
        """
        Suggest relevant analysis queries based on the database schema and business domain.

        Example:
        suggestions = DatabaseHelpers.suggest_analysis_queries('./sample_ecommerce.db')

        :param db_path: Path to the database file
        :return: List of suggested SQL queries for analysis
        """
        schema = DatabaseHelpers.get_database_schema(db_path)
        context = DatabaseHelpers.get_business_context(db_path)

        if not schema or "tables" not in schema:
            return ["-- Connect to database first to get suggestions"]

        suggestions = []
        tables = schema["tables"]
        domain = context.get("business_domain", "unknown")

        # Generic suggestions based on table structure
        for table_name, table_info in tables.items():
            columns = [col["name"] for col in table_info["columns"]]

            # Basic exploration queries
            suggestions.append(f"-- Explore {table_name} structure and data")
            suggestions.append(f"SELECT * FROM {table_name} LIMIT 10;")
            suggestions.append(f"SELECT COUNT(*) as total_rows FROM {table_name};")

            # Look for date columns for time-based analysis
            date_cols = [
                col for col in columns if any(word in col.lower() for word in ["date", "time", "created", "updated"])
            ]
            for date_col in date_cols:
                suggestions.append(
                    f"SELECT DATE({date_col}) as date, COUNT(*) FROM {table_name} GROUP BY DATE({date_col}) ORDER BY date;"
                )

        # Domain-specific suggestions
        if domain == "ecommerce":
            if "orders" in tables and "customers" in tables:
                suggestions.extend(
                    [
                        "-- E-commerce specific analysis",
                        "SELECT c.first_name, c.last_name, COUNT(o.order_id) as order_count, SUM(o.total_amount) as total_spent FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.customer_id ORDER BY total_spent DESC LIMIT 10;",
                        "SELECT DATE(order_date) as order_date, COUNT(*) as orders, SUM(total_amount) as revenue FROM orders GROUP BY DATE(order_date) ORDER BY order_date;",
                        "SELECT status, COUNT(*) as count, AVG(total_amount) as avg_amount FROM orders GROUP BY status;",
                    ]
                )

            if "products" in tables and "order_items" in tables:
                suggestions.extend(
                    [
                        "SELECT p.product_name, SUM(oi.quantity) as total_sold, SUM(oi.total_price) as revenue FROM products p JOIN order_items oi ON p.product_id = oi.product_id GROUP BY p.product_id ORDER BY revenue DESC LIMIT 10;",
                        "SELECT p.product_name, p.stock_quantity, CASE WHEN p.stock_quantity < 10 THEN 'Low Stock' WHEN p.stock_quantity < 50 THEN 'Medium Stock' ELSE 'High Stock' END as stock_level FROM products p ORDER BY p.stock_quantity;",
                    ]
                )

        return suggestions[:20]  # Limit to 20 suggestions

    @staticmethod
    def add_insight(db_path: str, insight: str) -> str:
        """
        Add a business insight about the database for future reference.

        Example:
        DatabaseHelpers.add_insight('./sample_ecommerce.db', 'Customer retention rate is 65% based on repeat orders')

        :param db_path: Path to the database file
        :param insight: Business insight to store
        :return: Confirmation message
        """
        DatabaseHelpers._load_cache_from_disk()
        cache_key = DatabaseHelpers._get_cache_key(db_path)

        if cache_key not in DatabaseHelpers._business_context_cache:
            DatabaseHelpers._business_context_cache[cache_key] = {
                "discovered_at": datetime.now().isoformat(),
                "business_domain": "unknown",
                "common_queries": [],
                "metrics": [],
                "insights": [],
            }

        context = DatabaseHelpers._business_context_cache[cache_key]
        context["insights"].append({"insight": insight, "added_at": datetime.now().isoformat()})

        # Limit to last 100 insights
        if len(context["insights"]) > 100:
            context["insights"] = context["insights"][-100:]

        DatabaseHelpers._save_cache_to_disk()

        return f"Insight added to database knowledge base for {db_path}"

    @staticmethod
    def clear_cache(db_path: Optional[str] = None) -> str:
        """
        Clear cached information for a specific database or all databases.

        Example:
        DatabaseHelpers.clear_cache('./sample_ecommerce.db')  # Clear specific database
        DatabaseHelpers.clear_cache()  # Clear all cached data

        :param db_path: Optional path to specific database, or None to clear all
        :return: Confirmation message
        """
        if db_path:
            cache_key = DatabaseHelpers._get_cache_key(db_path)
            DatabaseHelpers._schema_cache.pop(cache_key, None)
            DatabaseHelpers._business_context_cache.pop(cache_key, None)
            message = f"Cleared cache for {db_path}"
        else:
            DatabaseHelpers._schema_cache.clear()
            DatabaseHelpers._business_context_cache.clear()
            message = "Cleared all database cache"

        DatabaseHelpers._save_cache_to_disk()
        return message
