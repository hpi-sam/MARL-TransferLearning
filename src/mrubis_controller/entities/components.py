import enum
from typing import List

import torch

class Components(enum.Enum):
    AUTHENTICATION_SERVICE = 'Authentication Service'
    AVAILABILITY_ITEM_FILTER = 'Availability Item Filter'
    BID_AND_BUY_SERVICE = 'Bid and Buy Service'
    BUY_NOW_ITEM_FILTER = 'Buy Now Item Filter'
    CATEGORY_ITEM_FILTER = 'Category Item Filter'
    COMMENT_ITEM_FILTER = 'Comment Item Filter'
    FUTURE_SALES_ITEM_FILTER = 'Future Sales Item Filter'
    INVENTORY_SERVICE = 'Inventory Service'
    ITEM_MANAGEMENT_SERVICE = 'Item Management Service'
    LAST_SECOND_SALES_ITEM_FILTER = 'Last Second Sales Item Filter'
    PAST_SALES_ITEM_FILTER = 'Past Sales Item Filter'
    PERSISTENCE_SERVICE = 'Persistence Service'
    QUERY_SERVICE = 'Query Service'
    RECOMMENDATION_ITEM_FILTER = 'Recommendation Item Filter'
    REGION_ITEM_FILTER = 'Region Item Filter'
    REPUTATION_SERVICE = 'Reputation Service'
    SELLER_REPUTATION_ITEM_FILTER = 'Seller Reputation Item Filter'
    USER_MANAGEMENT_SERVICE = 'User Management Service'

    @classmethod
    def list(self) -> List['Components']:
        return sorted([component for component in Components])
    @classmethod
    def value_list(self) -> List[str]:
        return sorted([component.value for component in Components])
    @classmethod
    def from_tensor(self, tensor: torch.Tensor) -> 'Components':
        return Components.list()[tensor.argmax()]
    @classmethod
    def value_from_tensor(self, tensor: torch.Tensor) -> str:
        return Components.value_list()[tensor.argmax()]
