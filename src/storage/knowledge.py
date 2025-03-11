"""
Knowledge graph module.
Manages knowledge extraction from conversation history and graph operations.
"""

# Standard library imports
import json
import math
import os
import re
import uuid
from collections import defaultdict, OrderedDict
from datetime import datetime
from functools import lru_cache
from typing import List, Dict, Any, Optional, Set, Tuple, Union, Type

# Third-party imports
import networkx as nx
from networkx.algorithms import community
import numpy as np
from packaging import version
from loguru import logger
import logging

# Local imports
from ..core.exceptions import StorageError

class DataValidator:
    """Data validation and normalization class"""
    
    # Valid data types
    VALID_TYPES = {
        "string": str,
        "number": (int, float),
        "boolean": bool,
        "list": list,
        "dict": dict,
        "null": type(None)
    }
    
    # Regex patterns for special character and format checks
    PATTERNS = {
        "entity_id": re.compile(r"^[a-zA-Z0-9_-]+$"),
        "version": re.compile(r"^\d+\.\d+(\.\d+)?$"),
        "date": re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?([+-]\d{2}:?\d{2}|Z)?$")
    }
    
    # Case sensitive fields
    CASE_SENSITIVE_FIELDS = {"name", "title", "label", "description"}
    
    @classmethod
    def validate_entity_id(cls, entity_id: str) -> bool:
        """Validate entity ID format"""
        if not isinstance(entity_id, str):
            return False
        return bool(cls.PATTERNS["entity_id"].match(entity_id))
    
    @classmethod
    def validate_type(cls, value: Any, expected_type: Union[Type, Tuple[Type, ...]]) -> bool:
        """Validate value type"""
        return isinstance(value, expected_type)
    
    @classmethod
    def validate_properties(cls, properties: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate property values"""
        errors = []
        
        for key, value in properties.items():
            # Key format check
            if not cls.PATTERNS["entity_id"].match(key):
                errors.append(f"Invalid property key format: {key}")
                
            # Value type check
            if value is not None and not isinstance(value, (str, int, float, bool, list, dict)):
                errors.append(f"Invalid value type for {key}: {type(value)}")
                
            # Version format check
            if key == "version" and isinstance(value, str):
                if not cls.PATTERNS["version"].match(value):
                    errors.append(f"Invalid version format: {value}")
                    
            # Date format check
            if key.endswith("_at") and isinstance(value, str):
                if not cls.PATTERNS["date"].match(value):
                    errors.append(f"Invalid date format: {value}")
        
        return len(errors) == 0, errors
    
    @classmethod
    def normalize_string(cls, value: str, preserve_case: bool = False) -> str:
        """Normalize string values"""
        if preserve_case:
            return value.strip()
        return value.strip().lower()
    
    @classmethod
    def normalize_version(cls, value: Union[str, float]) -> str:
        """Normalize version values"""
        if isinstance(value, float):
            # Convert float to string (3.9 -> "3.9")
            value = f"{value:.2f}".rstrip('0').rstrip('.')
        return str(value)
    
    @classmethod
    def normalize_properties(cls, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize property values"""
        normalized = {}
        
        for key, value in properties.items():
            # String normalization
            if isinstance(value, str):
                if key == "version":
                    normalized[key] = cls.normalize_version(value)
                else:
                    # Preserve case for case-sensitive fields
                    preserve_case = key in cls.CASE_SENSITIVE_FIELDS
                    normalized[key] = cls.normalize_string(value, preserve_case)
            # List normalization
            elif isinstance(value, list):
                normalized[key] = [
                    cls.normalize_string(item, True) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                normalized[key] = value
                
        return normalized

class CacheManager:
    """Cache management class"""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize the cache manager
        
        Args:
            max_size: Maximum cache size
        """
        self.max_size = max_size
        self._entity_cache = OrderedDict()  # OrderedDict for LRU cache
        self._relation_cache = OrderedDict()  # Relation cache
        self._stats = {
            "entity_hits": 0,
            "entity_misses": 0,
            "relation_hits": 0,
            "relation_misses": 0,
            "cache_evictions": 0
        }
        
    def _evict_if_needed(self, cache: OrderedDict) -> None:
        """Remove the oldest item if cache is full"""
        if len(cache) >= self.max_size:
            cache.popitem(last=False)  # Remove the first added item (LRU)
            self._stats["cache_evictions"] += 1
            
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get entity information from cache"""
        if entity_id in self._entity_cache:
            # Cache hit - move item to the end (most recently used)
            self._stats["entity_hits"] += 1
            data = self._entity_cache.pop(entity_id)
            self._entity_cache[entity_id] = data
            return data.copy()
        self._stats["entity_misses"] += 1
        return None
        
    def set_entity(self, entity_id: str, data: Dict[str, Any]) -> None:
        """Add entity information to cache"""
        self._evict_if_needed(self._entity_cache)
        self._entity_cache[entity_id] = data.copy()
        
    def get_relation(self, source_id: str, target_id: str) -> Optional[Dict[str, Any]]:
        """Get relation information from cache"""
        key = (source_id, target_id)
        if key in self._relation_cache:
            # Cache hit
            self._stats["relation_hits"] += 1
            data = self._relation_cache.pop(key)
            self._relation_cache[key] = data
            return data.copy()
        self._stats["relation_misses"] += 1
        return None
        
    def set_relation(self, source_id: str, target_id: str, data: Dict[str, Any]) -> None:
        """Add relation information to cache"""
        self._evict_if_needed(self._relation_cache)
        self._relation_cache[(source_id, target_id)] = data.copy()
        
    def remove_entity(self, entity_id: str) -> None:
        """Remove entity from cache"""
        if entity_id in self._entity_cache:
            del self._entity_cache[entity_id]
            
        # Also remove related relations
        keys_to_remove = []
        for (source, target) in self._relation_cache:
            if source == entity_id or target == entity_id:
                keys_to_remove.append((source, target))
        for key in keys_to_remove:
            del self._relation_cache[key]
            
    def remove_relation(self, source_id: str, target_id: str) -> None:
        """Remove relation from cache"""
        key = (source_id, target_id)
        if key in self._relation_cache:
            del self._relation_cache[key]
            
    def clear(self) -> None:
        """Clear all cache"""
        self._entity_cache.clear()
        
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        stats = dict(self._stats)
        stats.update({
            "entity_cache_size": len(self._entity_cache),
            "relation_cache_size": len(self._relation_cache),
            "total_cache_size": len(self._entity_cache) + len(self._relation_cache)
        })
        return stats

class KnowledgeGraph:
    def __init__(self, 
                 storage_path: str = "./data/knowledge_graph.json",
                 similarity_threshold: float = 0.7,
                 name: str = "knowledge_graph",
                 cache_size: int = 1000):
        """
        Initialize knowledge graph
        
        Args:
            storage_path: File path where graph data will be saved
            similarity_threshold: Similarity threshold for node merging
            name: Name of the graph
            cache_size: Maximum cache size
        """
        self.storage_path = storage_path
        self.similarity_threshold = similarity_threshold
        self.graph = nx.DiGraph()
        self.min_confidence = 0.5
        self.validator = DataValidator()
        self.cache = CacheManager(max_size=cache_size)
        
        # Set graph properties
        self.graph.graph["created_at"] = datetime.now().isoformat()
        self.graph.graph["last_updated"] = datetime.now().isoformat()
        self.name = name
        
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "node_count": 0,
            "edge_count": 0,
        }
        
        # Create data directory
        self.data_dir = os.path.dirname(storage_path)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            
        # Load existing data
        self.load_graph()
        
    def _clear_cache(self):
        """Clear the cache"""
        self.cache.clear()

    def _update_graph_entity(self, entity_id: str, data: Dict[str, Any]) -> None:
        """Update entity data in the graph and cache"""
        # Update the graph
        self.graph.add_node(entity_id, **data)
        # Update cache
        self.cache.set_entity(entity_id, data)
        # Update timestamp
        self.graph.graph["last_updated"] = datetime.now().isoformat()

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get entity information
        
        Args:
            entity_id: Entity ID
            
        Returns:
            Entity information or None
        """
        # Check from cache
        entity = self.cache.get_entity(entity_id)
        if entity is not None:
            return entity.copy()
            
        # Check from graph
        if self.graph.has_node(entity_id):
            entity = dict(self.graph.nodes[entity_id])
            entity["id"] = entity_id
            self.cache.set_entity(entity_id, entity)
            return entity.copy()
            
        return None

    @lru_cache(maxsize=100)
    def _get_cached_path(self, source_id: str, target_id: str, max_depth: int) -> Optional[List[str]]:
        """Find and cache path between two entities"""
        try:
            paths = list(nx.all_simple_paths(self.graph, source_id, target_id, cutoff=max_depth))
            return min(paths, key=len) if paths else None
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _update_metadata(self):
        """Update metadata"""
        self.metadata["updated_at"] = datetime.now().isoformat()
        self.metadata["node_count"] = self.graph.number_of_nodes()
        self.metadata["edge_count"] = self.graph.number_of_edges()
        
    def add_entity(self,
                   entity_id: str,
                   entity_type: str,
                   properties: Dict[str, Any],
                   embedding: Optional[List[float]] = None) -> str:
        """
        Add a new entity to the graph
        
        Args:
            entity_id: Entity ID
            entity_type: Entity type
            properties: Entity properties
            embedding: Entity embedding vector (optional)
            
        Returns:
            ID of the added entity
        """
        # Prepare entity data
        entity_data = {
            "id": entity_id,
            "type": entity_type,
            "properties": properties.copy(),  # Copy properties
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "version": "1.0"
        }
        
        # Add embedding vector
        if embedding is not None:
            entity_data["embedding"] = embedding.copy()
        
        # Add to graph
        self.graph.add_node(entity_id, **entity_data)
        
        # Add to cache
        self.cache.set_entity(entity_id, entity_data)
        
        # Update metadata
        self._update_metadata()
        
        return entity_id
        
    def add_relation(self,
                    source_id: str,
                    target_id: str,
                    relation_type: str,
                    properties: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a relation between two entities
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relation_type: Relation type
            properties: Relation properties
        """
        # Validate entity IDs
        if not self.validator.validate_entity_id(source_id):
            raise ValueError(f"Invalid source_id format: {source_id}")
        if not self.validator.validate_entity_id(target_id):
            raise ValueError(f"Invalid target_id format: {target_id}")
            
        # Check if entities exist
        if not self.get_entity(source_id):
            raise ValueError(f"Source entity not found: {source_id}")
        if not self.get_entity(target_id):
            raise ValueError(f"Target entity not found: {target_id}")
            
        if properties is None:
            properties = {}
        else:
            # Validate and normalize properties
            is_valid, errors = self.validator.validate_properties(properties)
            if not is_valid:
                raise ValueError(f"Invalid properties: {', '.join(errors)}")
            properties = self.validator.normalize_properties(properties)
            
        # Prepare relation properties
        edge_data = {
            "type": relation_type,
            "created_at": datetime.now().isoformat(),
            "confidence": properties.get("confidence", 1.0),
            **properties
        }
        
        # Validate confidence score
        if not isinstance(edge_data["confidence"], (int, float)) or not 0 <= edge_data["confidence"] <= 1:
            raise ValueError("Confidence score must be a number between 0 and 1")
        
        # Add relation to graph
        self.graph.add_edge(source_id, target_id, **edge_data)
        # Add to cache
        self.cache.set_relation(source_id, target_id, edge_data)
        self.graph.graph["last_updated"] = datetime.now().isoformat()
        
    def get_relations(self, 
                     entity_id: str, 
                     relation_type: Optional[str] = None,
                     direction: str = "both") -> List[Dict[str, Any]]:
        """
        Get entity relations
        
        Args:
            entity_id: Entity ID
            relation_type: Relation type filter
            direction: Relation direction ("in", "out", "both")
            
        Returns:
            List of relations
        """
        relations = []
        
        # Check incoming relations
        if direction in ["in", "both"]:
            for source, target in self.graph.in_edges(entity_id):
                # Get relation information from cache or graph
                relation_data = self.cache.get_relation(source, target)
                if relation_data is None:
                    relation_data = dict(self.graph[source][target])
                    self.cache.set_relation(source, target, relation_data)
                
                if relation_type is None or relation_data["type"] == relation_type:
                    relations.append({
                        "source": source,
                        "target": target,
                        **relation_data
                    })
                    
        # Check outgoing relations
        if direction in ["out", "both"]:
            for source, target in self.graph.out_edges(entity_id):
                # Get relation information from cache or graph
                relation_data = self.cache.get_relation(source, target)
                if relation_data is None:
                    relation_data = dict(self.graph[source][target])
                    self.cache.set_relation(source, target, relation_data)
                
                if relation_type is None or relation_data["type"] == relation_type:
                    relations.append({
                        "source": source,
                        "target": target,
                        **relation_data
                    })
                    
        return relations
        
    def find_path(self,
                  source_id: str,
                  target_id: str,
                  max_depth: int = 3) -> Optional[List[Dict[str, Any]]]:
        """
        Find the shortest path between two entities
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            max_depth: Maximum depth
            
        Returns:
            Entity and relation information along the path
        """
        # Check cached path
        path = self._get_cached_path(source_id, target_id, max_depth)
        if path is None:
            return None
            
        result = []
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            
            # Get entity and relation information
            current_entity = self.get_entity(current)
            relation_data = self.cache.get_relation(current, next_node)
            if relation_data is None:
                relation_data = dict(self.graph[current][next_node])
                self.cache.set_relation(current, next_node, relation_data)
            
            result.append({
                "entity": current_entity,
                "relation": relation_data
            })
            
        # Add the last entity
        result.append({
            "entity": self.get_entity(path[-1]),
            "relation": None
        })
        
        return result
            
    def merge_entities(self, source_id: str, target_id: str, strategy: str = "keep_all") -> str:
        """
        Merge two entities
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            strategy: Merge strategy
            
        Returns:
            Merged entity ID
        """
        source = self.get_entity(source_id)
        target = self.get_entity(target_id)
        
        if not source or not target:
            raise ValueError("Source or target entity not found")
            
        # Merge properties
        merged_properties = target["properties"].copy()
        for key, value in source["properties"].items():
            if key not in merged_properties:
                merged_properties[key] = value
            elif strategy == "keep_all":
                if isinstance(value, list) and isinstance(merged_properties[key], list):
                    # Merge lists
                    merged_properties[key] = list(set(merged_properties[key] + value))
                elif isinstance(value, (int, float)) and isinstance(merged_properties[key], (int, float)):
                    # Average numerical values
                    merged_properties[key] = (merged_properties[key] + value) / 2
                else:
                    # For other types, keep source value
                    merged_properties[key] = value
                
        # Prepare new entity data
        merged_data = {
            "id": target_id,
            "type": target["type"],
            "properties": merged_properties,
            "created_at": min(source["created_at"], target["created_at"]),
            "updated_at": datetime.now().isoformat(),
            "version": str(float(target["version"]) + 0.1)
        }
        
        # Merge embedding vectors
        if "embedding" in source and "embedding" in target:
            merged_data["embedding"] = [
                (s + t) / 2 for s, t in zip(source["embedding"], target["embedding"])
            ]
        elif "embedding" in source:
            merged_data["embedding"] = source["embedding"].copy()
        elif "embedding" in target:
            merged_data["embedding"] = target["embedding"].copy()
            
        # Update graph
        self.graph.remove_node(source_id)
        self.graph.add_node(target_id, **merged_data)
        
        # Update cache
        self.cache.remove_entity(source_id)
        self.cache.set_entity(target_id, merged_data)
        
        # Update metadata
        self._update_metadata()
        
        return target_id
        
    def save(self, file_path: str) -> None:
        """
        Save graph to file
        
        Args:
            file_path: Path to save file
        """
        # Convert graph to JSON format
        graph_data = nx.node_link_data(self.graph, edges="links")
        
        # Fix entity properties
        for node in graph_data["nodes"]:
            if "id" not in node:
                node["id"] = node.get("name", str(uuid.uuid4()))
            if "properties" not in node:
                node["properties"] = {}
                
        data = {
            "graph": graph_data,
            "metadata": self.metadata
        }
        
        # Save to JSON file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
    def load(self, file_path: str) -> None:
        """
        Load graph from file
        
        Args:
            file_path: Path to file to load
        """
        # Read JSON file
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Fix entity properties
        for node in data["graph"]["nodes"]:
            if "id" not in node:
                node["id"] = node.get("name", str(uuid.uuid4()))
            if "properties" not in node:
                node["properties"] = {}
                
        # Load graph
        self.graph = nx.node_link_graph(data["graph"], edges="links")
        self.metadata = data["metadata"]
        
        # Clear cache and refill
        self.cache.clear()
        for node_id in self.graph.nodes:
            node_data = dict(self.graph.nodes[node_id])
            node_data["id"] = node_id
            self.cache.set_entity(node_id, node_data)
            
    def get_stats(self) -> Dict[str, Any]:
        """
        Get graph statistics
        
        Returns:
            Statistics
        """
        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "created_at": self.graph.graph.get("created_at"),
            "last_updated": self.graph.graph.get("last_updated")
        }
        
    def get_central_nodes(self, 
                         n_nodes: int = 5, 
                         metrics: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find the most central nodes
        
        Args:
            n_nodes: Number of nodes to return
            metrics: Centrality metrics to use ["degree", "betweenness", "eigenvector", "pagerank"]
            
        Returns:
            Most central nodes for each metric
        """
        if metrics is None:
            metrics = ["degree", "betweenness", "eigenvector", "pagerank"]
            
        results = {}
        
        for metric in metrics:
            if metric == "degree":
                scores = dict(self.graph.degree())
            elif metric == "betweenness":
                scores = nx.betweenness_centrality(self.graph)
            elif metric == "eigenvector":
                scores = nx.eigenvector_centrality(self.graph, max_iter=1000)
            elif metric == "pagerank":
                scores = nx.pagerank(self.graph)
            else:
                continue
                
            # Get nodes with highest scores
            top_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_nodes]
            
            # Prepare results
            results[metric] = [
                {
                    "node_id": node_id,
                    "score": score,
                    "properties": self.get_entity(node_id)
                }
                for node_id, score in top_nodes
            ]
            
        return results
        
    def detect_communities(self, 
                         algorithm: str = "louvain",
                         min_community_size: int = 2) -> List[Set[str]]:
        """
        Detect communities in the graph
        
        Args:
            algorithm: Algorithm to use ("louvain", "label_propagation", "greedy_modularity")
            min_community_size: Minimum community size
            
        Returns:
            Detected communities (sets of node IDs)
        """
        # Convert to undirected graph (for community detection)
        undirected = self.graph.to_undirected()
        
        if algorithm == "louvain":
            communities = community.louvain_communities(undirected)
        elif algorithm == "label_propagation":
            communities = community.label_propagation_communities(undirected)
        elif algorithm == "greedy_modularity":
            communities = community.greedy_modularity_communities(undirected)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
            
        # Minimum size filtering
        filtered_communities = [
            comm for comm in communities
            if len(comm) >= min_community_size
        ]
        
        return filtered_communities
        
    def analyze_community(self, 
                         community_nodes: Set[str]) -> Dict[str, Any]:
        """
        Bir topluluğu analiz et
        
        Args:
            community_nodes: Topluluk düğümlerinin ID'leri
            
        Returns:
            Topluluk analizi sonuçları
        """
        # Topluluk alt grafını al
        subgraph = self.graph.subgraph(community_nodes)
        undirected = subgraph.to_undirected()
        
        # Temel metrikleri hesapla
        density = nx.density(subgraph)
        
        # Çap hesaplama (yönsüz graf üzerinde)
        if nx.is_connected(undirected):
            diameter = nx.diameter(undirected)
        else:
            # Bağlı olmayan bileşenler için en büyük bileşenin çapını al
            components = list(nx.connected_components(undirected))
            if components:
                largest = max(components, key=len)
                largest_subgraph = undirected.subgraph(largest)
                diameter = nx.diameter(largest_subgraph)
            else:
                diameter = float('inf')
        
        # Düğüm tipleri dağılımı
        node_types = {}
        for node in community_nodes:
            entity = self.get_entity(node)
            if entity:
                node_type = entity.get("type", "unknown")
                node_types[node_type] = node_types.get(node_type, 0) + 1
                
        # En merkezi düğümü bul (yönsüz graf üzerinde)
        central_node = max(
            undirected.degree(),
            key=lambda x: x[1]
        )[0]
        
        # Bağlantı istatistikleri
        avg_in_degree = sum(d for _, d in subgraph.in_degree()) / len(community_nodes)
        avg_out_degree = sum(d for _, d in subgraph.out_degree()) / len(community_nodes)
        
        return {
            "size": len(community_nodes),
            "density": density,
            "diameter": diameter,
            "node_types": node_types,
            "central_node": self.get_entity(central_node),
            "avg_in_degree": avg_in_degree,
            "avg_out_degree": avg_out_degree
        }
        
    def find_paths(self,
                   source_id: str,
                   target_id: str,
                   max_depth: int = 3,
                   max_paths: int = 5) -> List[List[Dict[str, Any]]]:
        """
        İki varlık arasındaki tüm olası yolları bul ve puanla
        
        Args:
            source_id: Başlangıç varlık ID'si
            target_id: Hedef varlık ID'si
            max_depth: Maksimum derinlik
            max_paths: Maksimum yol sayısı
            
        Returns:
            Bulunan yollar (skor bazlı sıralı)
        """
        try:
            # Tüm yolları bul
            paths = list(nx.all_simple_paths(self.graph, source_id, target_id, cutoff=max_depth))
            if not paths:
                return []
                
            # Her yol için skor hesapla
            scored_paths = []
            for path in paths:
                path_data = []
                path_score = 0
                
                for i in range(len(path) - 1):
                    current = path[i]
                    next_node = path[i + 1]
                    
                    # Düğüm ve kenar verilerini al
                    current_node = self.get_entity(current)
                    edge_data = dict(self.graph[current][next_node])
                    
                    # Yol skorunu güncelle
                    path_score += self._calculate_edge_weight(current_node, edge_data)
                    
                    # Yol verilerini ekle
                    path_data.append({
                        "entity": current_node,
                        "relation": edge_data
                    })
                
                # Son düğümü ekle
                path_data.append({
                    "entity": self.get_entity(path[-1]),
                    "relation": None
                })
                
                scored_paths.append((path_data, path_score))
            
            # Skorlara göre sırala ve en iyi yolları döndür
            scored_paths.sort(key=lambda x: x[1], reverse=True)
            return [path for path, _ in scored_paths[:max_paths]]
            
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
            
    def _calculate_edge_weight(self,
                             node: Dict[str, Any],
                             edge: Dict[str, Any]) -> float:
        """
        Calculate edge weight
        
        Args:
            node: Node properties
            edge: Edge properties
            
        Returns:
            Edge weight (between 0-1)
        """
        weight = 1.0
        
        # Weight based on edge type
        if "type" in edge:
            if edge["type"] in ["is_a", "part_of"]:
                weight *= 1.0
            elif edge["type"] in ["related_to", "similar_to"]:
                weight *= 0.8
            else:
                weight *= 0.6
                
        # Weight based on confidence score
        if "confidence" in edge:
            weight *= float(edge["confidence"])
            
        # Time-based weight (newer relationships are more important)
        if "created_at" in edge:
            age = (datetime.now() - datetime.fromisoformat(edge["created_at"])).days
            time_weight = 1.0 / (1.0 + (age / 365))  # half weight for 1 year old
            weight *= time_weight
            
        return weight
        
    def suggest_paths(self,
                     source_id: str,
                     target_type: Optional[str] = None,
                     min_confidence: float = 0.5,
                     max_suggestions: int = 5) -> List[Dict[str, Any]]:
        """
        Generate possible path suggestions starting from an entity
        
        Args:
            source_id: Source entity ID
            target_type: Target entity type (optional)
            min_confidence: Minimum confidence score
            max_suggestions: Maximum number of suggestions
            
        Returns:
            Suggested paths and their explanations
        """
        suggestions = []
        source_node = self.get_entity(source_id)
        if not source_node:
            return []
            
        # Scan paths with BFS
        visited = {source_id}
        queue = [(source_id, [], 1.0)]  # (node_id, path, confidence)
        
        while queue and len(suggestions) < max_suggestions:
            current_id, path, confidence = queue.pop(0)
            current = self.get_entity(current_id)
            
            # Add suggestion if it matches target type and has sufficient confidence score
            if current_id != source_id and confidence >= min_confidence:
                if target_type is None or current.get("type") == target_type:
                    suggestion = {
                        "target": current,
                        "path": path + [current],
                        "confidence": confidence,
                        "explanation": self._generate_path_explanation(path + [current])
                    }
                    suggestions.append(suggestion)
                    continue
                    
            # Check neighboring nodes
            for _, neighbor, edge in self.graph.out_edges(current_id, data=True):
                if neighbor not in visited:
                    visited.add(neighbor)
                    edge_confidence = edge.get("confidence", 0.8)
                    new_confidence = confidence * edge_confidence
                    if new_confidence >= min_confidence:
                        queue.append((
                            neighbor,
                            path + [current],
                            new_confidence
                        ))
                        
        return suggestions
        
    def _generate_path_explanation(self, path: List[Dict[str, Any]]) -> str:
        """
        Generate natural language explanation for a path
        
        Args:
            path: Nodes along the path
            
        Returns:
            Path explanation
        """
        if not path:
            return ""
            
        explanation = []
        for i, node in enumerate(path):
            name = node.get("name", node.get("id", ""))
            if i == 0:
                explanation.append(f"{name}")
            else:
                prev = path[i-1]
                prev_name = prev.get("name", prev.get("id", ""))
                relation = self.graph[prev["id"]][node["id"]].get("type", "related")
                explanation.append(f"{relation} {name}")
                
        return " -> ".join(explanation)
        
    def update_entity(self, entity_id: str, new_properties: Dict[str, Any],
                     confidence: float = 1.0, merge_strategy: str = "replace") -> Dict[str, Any]:
        """Update entity properties"""
        # Validate entity_id
        if not self.validator.validate_entity_id(entity_id):
            raise ValueError(f"Invalid entity_id format: {entity_id}")
            
        # Validate confidence score
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError("Confidence score must be a number between 0 and 1")
            
        # Validate merge strategy
        valid_strategies = ["replace", "weighted_average"]
        if merge_strategy not in valid_strategies:
            raise ValueError(f"Invalid merge strategy. Must be one of: {valid_strategies}")
            
        entity = self.get_entity(entity_id)
        if not entity:
            raise ValueError(f"Entity {entity_id} not found")
            
        # Validate and normalize new properties
        is_valid, errors = self.validator.validate_properties(new_properties)
        if not is_valid:
            raise ValueError(f"Invalid properties: {', '.join(errors)}")
        new_properties = self.validator.normalize_properties(new_properties)

        # Update timestamp
        entity["last_updated"] = datetime.now().isoformat()

        # Update properties
        for key, new_value in new_properties.items():
            if merge_strategy == "weighted_average":
                old_value = entity.get(key, new_value)
                old_confidence = entity.get(f"{key}_confidence", 0.5)  # Default old confidence is 0.5
                
                # For version numbers, use version comparison
                if key == "version":
                    # Convert to strings for version comparison
                    new_ver = self.validator.normalize_version(new_value)
                    old_ver = self.validator.normalize_version(old_value)
                    
                    logger.debug(f"Comparing versions: {new_ver} vs {old_ver}")
                    logger.debug(f"New version parsed: {version.parse(new_ver)}")
                    logger.debug(f"Old version parsed: {version.parse(old_ver)}")
                    
                    # If new version is higher, use it
                    if version.parse(new_ver) > version.parse(old_ver):
                        logger.debug("New version is higher")
                        entity[key] = new_ver  # Store as string
                        # Keep the higher confidence score
                        entity[f"{key}_confidence"] = max(confidence, old_confidence)
                    else:
                        logger.debug("Old version is higher or equal")
                        # Keep old version if it's higher
                        entity[key] = old_ver  # Store as string
                        entity[f"{key}_confidence"] = old_confidence
                elif isinstance(new_value, (int, float)):
                    # For other numeric values, use weighted average
                    total_confidence = confidence + old_confidence
                    weighted_value = (new_value * confidence + old_value * old_confidence) / total_confidence
                    entity[key] = weighted_value
                    entity[f"{key}_confidence"] = max(confidence, old_confidence)
                else:
                    entity[key] = new_value
                    entity[f"{key}_confidence"] = confidence
            else:  # replace strategy
                entity[key] = new_value
                entity[f"{key}_confidence"] = confidence
                
            entity[f"{key}_updated"] = datetime.now().isoformat()

        # Update graph and cache
        self._update_graph_entity(entity_id, entity)
        
        # Update metadata
        self._update_metadata()
        
        return entity
        
    def update_relation(self,
                       source_id: str,
                       target_id: str,
                       relation_type: str,
                       new_confidence: float,
                       min_update_interval: int = 1) -> Dict[str, Any]:
        """
        Update relation confidence score
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relation_type: Relation type
            new_confidence: New confidence score
            min_update_interval: Minimum update interval (days)
            
        Returns:
            Updated relation information
        """
        if not self.graph.has_edge(source_id, target_id):
            raise ValueError(f"Relation not found: {source_id} -> {target_id}")
            
        # Get relation information from cache or graph
        edge_data = self.cache.get_relation(source_id, target_id)
        if edge_data is None:
            edge_data = dict(self.graph[source_id][target_id])
        
        # Check relation type
        if edge_data.get("type") != relation_type:
            raise ValueError(f"Relation type mismatch: {edge_data.get('type')} != {relation_type}")
            
        # Check last update time
        last_updated = edge_data.get("last_updated")
        if last_updated:
            days_since_update = (datetime.now() - datetime.fromisoformat(last_updated)).days
            if days_since_update < min_update_interval:
                return edge_data
        
        # Update confidence score (exponential moving average)
        old_confidence = edge_data.get("confidence", 0.5)
        alpha = 0.7  # Weight given to new value
        updated_confidence = alpha * new_confidence + (1 - alpha) * old_confidence
        
        # Update relation
        edge_data.update({
            "confidence": updated_confidence,
            "last_updated": datetime.now().isoformat()
        })
        
        # Update graph and cache
        self.graph[source_id][target_id].update(edge_data)
        self.cache.set_relation(source_id, target_id, edge_data)
        self.graph.graph["last_updated"] = datetime.now().isoformat()
        
        # Update metadata
        self._update_metadata()
        
        return edge_data.copy()
        
    def detect_conflicts(self, entity_id: str) -> List[Dict[str, Any]]:
        """Detect conflicts for an entity."""
        entity = self.get_entity(entity_id)
        if not entity:
            return []

        conflicts = []

        # Check for low confidence properties
        for key in entity.keys():
            if key.endswith("_confidence"):
                prop_name = key[:-11]  # Remove "_confidence" suffix
                if entity[key] < self.min_confidence:
                    conflicts.append({
                        "type": "low_confidence",
                        "property": prop_name,
                        "confidence": entity[key]
                    })

        # Check for conflicting relations
        relations = list(self.get_relations(entity_id))
        if relations:
            by_type = defaultdict(list)
            for rel in relations:
                rel_type = rel.get("type", "default")
                if rel.get("confidence", 1.0) < self.min_confidence:
                    by_type[rel_type].append(rel)

            for rel_type, rels in by_type.items():
                if len(rels) > 0:  # Any low confidence relation is a conflict
                    conflicts.append({
                        "type": "conflicting_relations",
                        "relation_type": rel_type,
                        "relations": rels
                    })

        return conflicts
        
    def resolve_conflicts(self, entity_id: str) -> List[Dict[str, Any]]:
        """Resolve conflicts for an entity by removing low confidence data."""
        conflicts = self.detect_conflicts(entity_id)
        if not conflicts:
            return []

        entity = self.get_entity(entity_id)
        resolved = []

        # Remove low confidence properties
        properties_to_remove = []
        for conflict in conflicts:
            if conflict["type"] == "low_confidence":
                prop_name = conflict["property"]
                properties_to_remove.extend([
                    prop_name,
                    f"{prop_name}_confidence",
                    f"{prop_name}_updated"
                ])
                resolved.append({
                    "type": "property_removed",
                    "property": prop_name,
                    "confidence": conflict["confidence"]
                })

        # Actually remove the properties
        for key in properties_to_remove:
            if key in entity:
                del entity[key]

        # Update the graph
        self._update_graph_entity(entity_id, entity)
        
        # Update metadata
        self._update_metadata()

        # Remove low confidence relations
        for conflict in conflicts:
            if conflict["type"] == "conflicting_relations":
                for relation in conflict["relations"]:
                    self.graph.remove_edge(relation["source"], relation["target"])
                    resolved.append({
                        "type": "relation_removed",
                        "relation": relation
                    })

        return resolved
        
    def add_document_info(self, doc_id: str, file_info: Dict[str, Any]) -> None:
        """
        Add document information to the graph
        
        Args:
            doc_id: Document ID
            file_info: Document information
        """
        try:
            # Add document node
            self.graph.add_node(
                doc_id,
                type="document",
                label=file_info.get("filename", "Unnamed Document"),
                properties=file_info,
                created_at=datetime.now().isoformat()
            )
            
            # Update graph
            self.graph.graph["last_updated"] = datetime.now().isoformat()
            logging.info(f"Document {doc_id} added to knowledge graph")
            
            # Save changes
            self.save_graph()
            
        except Exception as e:
            logging.error(f"Error adding document to knowledge graph: {str(e)}")
            # Not raising error so the API can continue working
            # raise StorageError(f"Knowledge graph error: {str(e)}")
    
    def save_graph(self):
        """
        Save the knowledge graph to file
        """
        try:
            # Create directory (if it doesn't exist)
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            # Save graph in JSON format
            data = {
                "metadata": self.metadata,
                "nodes": [
                    {**self.graph.nodes[node], "id": node}
                    for node in self.graph.nodes
                ],
                "edges": [
                    {
                        "source": u,
                        "target": v,
                        **self.graph.edges[u, v]
                    }
                    for u, v in self.graph.edges
                ]
            }
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            logging.info(f"Knowledge graph saved to {self.storage_path}")
            
        except Exception as e:
            logging.error(f"Error saving knowledge graph: {e}")
    
    def load_graph(self):
        """
        Load knowledge graph from file
        """
        if not os.path.exists(self.storage_path):
            logging.info(f"No existing knowledge graph found at {self.storage_path}")
            return
            
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Load metadata
            self.metadata = data.get("metadata", {})
            
            # Clear graph
            self.graph.clear()
            
            # Load nodes
            for node_data in data.get("nodes", []):
                node_id = node_data.pop("id")
                self.graph.add_node(node_id, **node_data)
                
            # Load edges
            for edge_data in data.get("edges", []):
                source = edge_data.pop("source")
                target = edge_data.pop("target")
                self.graph.add_edge(source, target, **edge_data)
                
            logging.info(f"Knowledge graph loaded from {self.storage_path} with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            
        except Exception as e:
            logging.error(f"Error loading knowledge graph: {e}")
            # Create a new graph
            self.graph = nx.DiGraph() 