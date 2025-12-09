
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum


# cloth item category
class ItemCategory(Enum):
    
    TOP = "top"
    BOTTOM = "bottom"
    SHOE = "shoe"
    ACCESSORY = "accessory"

# formality
class FormalityLevel(Enum):

    CASUAL = 1
    SMART_CASUAL = 2
    BUSINESS_CASUAL = 3
    FORMAL = 4
    ATHLETIC = 5


# one single cloth item

@dataclass
class ClothingItem:
    item_id: str
    category: ItemCategory
    image_path: str
    embedding: Optional[List[float]] = None
    
    attributes: Set[str] = None
    colors: Set[str] = None
    patterns: Set[str] = None
    formality: Optional[FormalityLevel] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = set()
        if self.colors is None:
            self.colors = set()
        if self.patterns is None:
            self.patterns = set()


#  outfit combination

@dataclass
class Outfit:

    outfit_id: str
    top: ClothingItem
    bottom: ClothingItem
    shoes: ClothingItem
    accessories: List[ClothingItem] = None
    
    def __post_init__(self):
        if self.accessories is None:
            self.accessories = []
    
    def to_dict(self) -> Dict:
        """Convert outfit to dictionary format"""
        return {
            "outfit_id": self.outfit_id,
            "top_id": self.top.item_id,
            "bottom_id": self.bottom.item_id,
            "shoes_id": self.shoes.item_id,
            "accessory_ids": [acc.item_id for acc in self.accessories]
        }


# restriction rules- check diagram

class CompatibilityRules:
  
    # used AI for generating rules- not into fashion so i don' tl
    #know what is good style
    # this is just example always can change
    CONFLICTING_PATTERNS = [
        {"stripe", "plaid"},
        {"floral", "animal print"},
        {"paisley", "geometric"}
    ]
    
    # Color combinations- this is just example always can change
    COMPLEMENTARY_COLORS = {
        "black": {"white", "grey", "beige", "navy"},
        "white": {"black", "navy", "grey", "beige", "any"},
        "navy": {"white", "beige", "grey", "black"},
        "beige": {"white", "navy", "brown", "black"},
        "grey": {"white", "black", "navy", "pink", "blue"}
    }
    
    @staticmethod
    def check_pattern_compatibility(item1: ClothingItem, item2: ClothingItem) -> bool:
        #find two clors matches as rules
        patterns1 = item1.patterns
        patterns2 = item2.patterns
        
        # If both items have no patterns, it is good match
        if not patterns1 and not patterns2:
            return True
        
        # If only one has a pattern,  still good match
        if not patterns1 or not patterns2:
            return True
        
        # one for not matching
        for conflict_set in CompatibilityRules.CONFLICTING_PATTERNS:
            if any(p in patterns1 for p in conflict_set) and \
               any(p in patterns2 for p in conflict_set):
                return False
        
        return True
    
    @staticmethod
    def check_color_compatibility(item1: ClothingItem, item2: ClothingItem) -> bool:
        """Check if colors of two items work together"""
        colors1 = item1.colors
        colors2 = item2.colors
        
        if not colors1 or not colors2:
            return True
        
    
        for c1 in colors1:
            for c2 in colors2:
                if c1 in CompatibilityRules.COMPLEMENTARY_COLORS:
                    if c2 in CompatibilityRules.COMPLEMENTARY_COLORS[c1] or \
                       "any" in CompatibilityRules.COMPLEMENTARY_COLORS[c1]:
                        return True
        
        return True  
    
    @staticmethod
    # this part checks formality eg. sports, business casual etc
    def check_formality_compatibility(item1: ClothingItem, item2: ClothingItem) -> bool:
        """Check if formality levels match"""
        if not item1.formality or not item2.formality:
            return True
        
        return abs(item1.formality.value - item2.formality.value) <= 1
    
    @staticmethod
    def check_compatibility(item1: ClothingItem, item2: ClothingItem) -> bool:
        """Overall compatibility check"""
        return (
            CompatibilityRules.check_pattern_compatibility(item1, item2) and
            CompatibilityRules.check_color_compatibility(item1, item2) and
            CompatibilityRules.check_formality_compatibility(item1, item2)
        )


class CombinationGenerator:
    """
    Layer 2: Generate outfit combinations using sequential ANN search
    """
    #faiss1- faiss 
    def __init__(self, ann_search: faiss1):

        self.ann_search = ann_search
        self.rules = CompatibilityRules()
    
           #anchor_item: Starting item (e.g., a favorite shirt)
            #num_candidates: Number of outfit combinations to generate
            #context: Optional context (occasion, weather, etc.)
    def generate_outfits(
        self,
        anchor_item: ClothingItem,
        num_candidates: int = 50,
        context: Optional[Dict] = None
    ) -> List[Outfit]:
    
        outfits = []
        
        # search order : top-botton-shoe 
        if anchor_item.category == ItemCategory.TOP:
            outfits = self._generate_from_top(anchor_item, num_candidates, context)
        elif anchor_item.category == ItemCategory.BOTTOM:
            outfits = self._generate_from_bottom(anchor_item, num_candidates, context)
        elif anchor_item.category == ItemCategory.SHOE:
            outfits = self._generate_from_shoes(anchor_item, num_candidates, context)
        
        return outfits
    
    def _generate_from_top(
        self,
        top: ClothingItem,
        num_candidates: int,
        context: Optional[Dict]
    ) -> List[Outfit]:
        outfits = []
        
        # Step 1: Find compatible bottoms using ANN search
        bottom_filters = self._create_filters(top, context)
        candidate_bottoms = self.ann_search.search_similar(
            query_embedding=top.embedding,
            category=ItemCategory.BOTTOM,
            k=num_candidates * 2,  # Get more candidates for filtering
            filters=bottom_filters
        )
        
        # Step 2: For each bottom, find compatible shoes
        for bottom in candidate_bottoms:
            # Check compatibility between top and bottom
            if not self.rules.check_compatibility(top, bottom):
                continue
            
            # Find shoes that work with both top and bottom
            shoe_filters = self._create_filters_for_shoes(top, bottom, context)
            candidate_shoes = self.ann_search.search_similar(
                query_embedding=bottom.embedding,  # Use bottom as query for shoes
                category=ItemCategory.SHOE,
                k=10,
                filters=shoe_filters
            )
            
            # Step 3: Create outfit combinations
            for shoes in candidate_shoes:
                # Check if shoes work with both top and bottom
                if not self.rules.check_compatibility(bottom, shoes):
                    continue
                if not self.rules.check_compatibility(top, shoes):
                    continue
                
                # Create outfit
                outfit_id = f"outfit_{len(outfits):04d}"
                outfit = Outfit(
                    outfit_id=outfit_id,
                    top=top,
                    bottom=bottom,
                    shoes=shoes
                )
                outfits.append(outfit)
                
                # Stop if we have enough candidates
                if len(outfits) >= num_candidates:
                    return outfits
        
        return outfits
    
    def _generate_from_bottom(
        self,
        bottom: ClothingItem,
        num_candidates: int,
        context: Optional[Dict]
    ) -> List[Outfit]:
        outfits = []
        
        # Find compatible tops
        top_filters = self._create_filters(bottom, context)
        candidate_tops = self.ann_search.search_similar(
            query_embedding=bottom.embedding,
            category=ItemCategory.TOP,
            k=num_candidates * 2,
            filters=top_filters
        )
        
        # For each top, find shoes
        for top in candidate_tops:
            if not self.rules.check_compatibility(top, bottom):
                continue
            
            shoe_filters = self._create_filters_for_shoes(top, bottom, context)
            candidate_shoes = self.ann_search.search_similar(
                query_embedding=bottom.embedding,
                category=ItemCategory.SHOE,
                k=10,
                filters=shoe_filters
            )
            
            for shoes in candidate_shoes:
                if not self.rules.check_compatibility(bottom, shoes):
                    continue
                if not self.rules.check_compatibility(top, shoes):
                    continue
                
                outfit_id = f"outfit_{len(outfits):04d}"
                outfit = Outfit(
                    outfit_id=outfit_id,
                    top=top,
                    bottom=bottom,
                    shoes=shoes
                )
                outfits.append(outfit)
                
                if len(outfits) >= num_candidates:
                    return outfits
        
        return outfits
    
    def _generate_from_shoes(
        self,
        shoes: ClothingItem,
        num_candidates: int,
        context: Optional[Dict]
    ) -> List[Outfit]:
        """Generate outfits starting from shoes"""
        outfits = []
        
        # Find bottoms that work with the shoes
        bottom_filters = self._create_filters(shoes, context)
        candidate_bottoms = self.ann_search.search_similar(
            query_embedding=shoes.embedding,
            category=ItemCategory.BOTTOM,
            k=num_candidates * 2,
            filters=bottom_filters
        )
        
        for bottom in candidate_bottoms:
            if not self.rules.check_compatibility(shoes, bottom):
                continue
            
            # Find tops
            top_filters = self._create_filters(bottom, context)
            candidate_tops = self.ann_search.search_similar(
                query_embedding=bottom.embedding,
                category=ItemCategory.TOP,
                k=10,
                filters=top_filters
            )
            
            for top in candidate_tops:
                if not self.rules.check_compatibility(top, bottom):
                    continue
                if not self.rules.check_compatibility(top, shoes):
                    continue
                
                outfit_id = f"outfit_{len(outfits):04d}"
                outfit = Outfit(
                    outfit_id=outfit_id,
                    top=top,
                    bottom=bottom,
                    shoes=shoes
                )
                outfits.append(outfit)
                
                if len(outfits) >= num_candidates:
                    return outfits
        
        return outfits
    
    def _create_filters(
        self,
        reference_item: ClothingItem,
        context: Optional[Dict]
    ) -> Dict:
        """Create filters for ANN search based on reference item and context"""
        filters = {}
        
        # Formality filter
        if reference_item.formality:
            filters["formality"] = reference_item.formality
        
        # Context-based filters
        if context:
            if "occasion" in context:
                filters["occasion"] = context["occasion"]
            if "weather" in context:
                filters["weather"] = context["weather"]
        
        return filters
    
    def _create_filters_for_shoes(
        self,
        top: ClothingItem,
        bottom: ClothingItem,
        context: Optional[Dict]
    ) -> Dict:
        """Create specific filters for shoe search based on top and bottom"""
        filters = {}
        
        # Use the more formal item to determine shoe formality
        if top.formality and bottom.formality:
            filters["formality"] = max(top.formality, bottom.formality, key=lambda x: x.value)
        elif top.formality:
            filters["formality"] = top.formality
        elif bottom.formality:
            filters["formality"] = bottom.formality
        
        if context:
            if "occasion" in context:
                filters["occasion"] = context["occasion"]
        
        return filters