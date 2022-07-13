from dataclasses import dataclass
import dataclasses
from typing import Dict, List

import torch
from entities.components import Components
from entities.component_failure import ComponentFailure
from entities.shop import Shop


@dataclass
class ComponentObservation:
    component_utility: float
    failure_name: ComponentFailure
    importance: float
    criticality: float
    reliability: float
    connectivity: float
    provided_interface: float
    required_interface: float
    adt: float
    perf_max: float
    sat_point: float
    replica: float
    request: float
    shop_utility: float
    system_utility: float
    root_issue: bool

    @staticmethod
    def from_dict(d: Dict) -> 'ComponentObservation':
        return ComponentObservation(
            failure_name=ComponentFailure[d['failure_name'].upper()],
            root_issue='root_issue' in d,
            **{k: float(v) for k, v in d.items() if k not in ['failure_name', 'uid', 'root_issue']}
        )

    def encode_to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.component_utility,
            float(self.failure_name.value),
            self.importance,
            self.criticality,
            self.reliability,
            self.connectivity,
            self.provided_interface,
            self.required_interface,
            self.adt,
            self.perf_max,
            self.sat_point,
            self.replica,
            self.request,
            self.shop_utility,
            self.system_utility
        ], dtype=torch.float32)

    def encode_to_dict(self):
        return {
            'component_utility': self.component_utility,
            'failure_name': self.failure_name.name if self.failure_name is not ComponentFailure.NONE else 'None',
            'importance': self.importance,
            'criticality': self.criticality,
            'reliability': self.reliability,
            'connectivity': self.connectivity,
            'provided_interface': self.provided_interface,
            'required_interface': self.required_interface,
            'adt': self.adt,
            'perf_max': self.perf_max,
            'sat_point': self.sat_point,
            'replica': self.replica,
            'request': self.request,
            'shop_utility': self.shop_utility,
            'system_utility': self.system_utility,
            'root_issue': self.root_issue,
        }


@dataclass
class ShopObservation:
    components: Dict[Components, ComponentObservation]
    shop_utility: float

    @staticmethod
    def from_dict(d: Dict) -> 'ShopObservation':
        component_keys = sorted(list(d.keys()))
        return ShopObservation(
            components={component_name: ComponentObservation.from_dict(
                d[component_name]) for component_name in component_keys},
            shop_utility=float(next(iter(d.values()))['shop_utility'])
        )

    def encode_to_tensor(self) -> torch.Tensor:
        return torch.concat([component.encode_to_tensor() for  _, component in sorted(list(self.components.items()), key=lambda x: x[0])])


@dataclass
class SystemObservation:
    shops: Dict[Shop, ShopObservation]
    system_utility: float

    @staticmethod
    def from_dict(d: Dict) -> 'SystemObservation':
        return SystemObservation(
            shops={shop_name: ShopObservation.from_dict(
                shop_obs) for shop_name, shop_obs in d.items()},
            system_utility=float(
                next(iter(next(iter(d.values())).values()))['system_utility'])
        )

    def encode_to_tensors(self) -> Dict[str, torch.Tensor]:
        return {shop_name: shop.encode_to_tensor() for shop_name, shop in self.shops.items()}

    def shop_utilites(self) -> Dict[Shop, float]:
        return {shop: shop_obs.shop_utility for shop, shop_obs in self.shops.items()}


@dataclass
class Action:
    shop: Shop
    component: Components
    predicted_utility: float

    def to_sendable_json(self) -> Dict:
        return dataclasses.asdict(self)


@dataclass
class RawAction:
    action: Action
    action_tensor: torch.Tensor
    expected_utility_tensor: torch.Tensor
    action_index: int
    observation_tensor: torch.Tensor

    def to_sendable_json(self) -> Dict:
        return self.shop.to_sendable_json()
