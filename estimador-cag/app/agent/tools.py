"""Strict tool contracts and evidence-backed budget arithmetic."""

from decimal import Decimal, ROUND_HALF_UP
from statistics import median

from pydantic import BaseModel, ConfigDict, Field


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Filters(Arguments):
    sector: str | None = Field(description="Exact historical client sector, or null for all sectors.")


class SearchBudgets(Arguments):
    component: str = Field(min_length=1, description="Stable, distinct name of the component being priced.")
    query: str = Field(min_length=1, max_length=2000, description="English search terms for this component.")
    filters: Filters | None = Field(description="Use null unless the transcript requires a sector restriction.")
    purpose: str = Field(min_length=1, description="A short user-facing description of the search objective.")


class Component(Arguments):
    name: str = Field(min_length=1, description="Reuse the exact component name used to search.")
    reference_ids: list[str] = Field(description="Distinct matching IDs from searches for this component, in order.")
    reference_amounts: list[int] = Field(description="Recorded engineer-hours for each ID in the same order.")


class CalculateEstimate(Arguments):
    components: list[Component] = Field(min_length=1, description="Components to price together.")
    purpose: str = Field(min_length=1, description="A short user-facing description of what is being calculated.")


def tool(name, contract, description):
    return {"type": "function", "name": name, "description": description,
            "parameters": contract.model_json_schema(), "strict": True}


TOOLS = [
    tool("search_budgets", SearchBudgets,
         "Find comparable historical work for ONE distinct component. Returns IDs, scope and reference hours. "
         "Search again with different terms if the scope is a poor match."),
    tool("calculate_estimate", CalculateEstimate,
         "Price one or more components from previously returned references. Use empty lists when no "
         "reference fits. Submit all scoped components together for the final total."),
]


def calculate_estimate(args, evidence, hourly_rate):
    """Use the median of matching references, with the course's 15% contingency."""
    rows = []
    names = set()
    for component in args.components:
        name = component.name
        ids = component.reference_ids
        hours = component.reference_amounts
        if name in names or name not in evidence:
            raise ValueError("Use a distinct component name previously passed to search_budgets.")
        names.add(name)
        if len(set(ids)) != len(ids) or len(ids) != len(hours):
            raise ValueError("Provide one amount per distinct reference ID.")
        for ref, amount in zip(ids, hours):
            if ref not in evidence[name] or evidence[name][ref] != amount:
                raise ValueError(f"Reference ID or hours do not match the search evidence for {name}.")
        estimate = (median(list(map(Decimal, hours))) * Decimal("1.15")).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP) if hours else Decimal(0)
        cost = (estimate * hourly_rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        rows.append({"name": name, "reference_ids": ids, "reference_amounts": hours,
                     "estimated_hours": float(estimate), "cost_eur": float(cost), "unbudgeted": not ids})
    return {"components": rows, "estimated_hours": round(sum(r["estimated_hours"] for r in rows), 1),
            "cost_eur": round(sum(r["cost_eur"] for r in rows), 2), "hourly_rate_eur": float(hourly_rate),
            "contingency_percent": 15, "complete": all(not r["unbudgeted"] for r in rows)}
