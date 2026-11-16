"""Explicit experiment graph configuration; never loaded by the application."""
import os
from pathlib import Path
import re
import secrets

ENV = Path(__file__).parent / '.env.neo4j.local'


def provision(path=ENV):
    """Create once. Existing passwords must survive container/CLI restarts."""
    if path.exists():
        return False
    with path.open('x', encoding='utf-8') as stream:
        stream.write('NEO4J_URI=bolt://127.0.0.1:7687\nNEO4J_USERNAME=neo4j\n'
                     f'NEO4J_PASSWORD={secrets.token_hex(24)}\nNEO4J_DATABASE=neo4j\n')
    return True


def configure(workspace, path=ENV):
    from dotenv import dotenv_values
    if not re.fullmatch(r'finsight_lightrag_[a-z0-9_]+', workspace):
        raise ValueError('Neo4j requires an isolated finsight_lightrag_* workspace')
    local = dotenv_values(path, interpolate=False) if path.exists() else {}
    override = os.environ.get('NEO4J_WORKSPACE', local.get('NEO4J_WORKSPACE'))
    if override and override != workspace:
        raise ValueError('NEO4J_WORKSPACE conflicts with this index; refusing cross-index writes')
    values = {key: os.environ.get(key, local.get(key)) for key in
              ['NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD', 'NEO4J_DATABASE']}
    if not all(values.values()):
        raise ValueError('Incomplete Neo4j config; run neo4j_experiment.py init first')
    # LightRAG's official Neo4j driver reads process environment. This module is
    # CLI-only: validate the complete configuration before changing any values.
    os.environ.update(values)
    return values
