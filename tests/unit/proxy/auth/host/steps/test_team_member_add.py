import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Final

import pytest
from fastapi import HTTPException

from litellm.proxy.auth.host.steps.team_member_add import add_team_member
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import ProxyLogging
from tests.unit.proxy.rust_auth_stores import RustAuthStores

TEAM: Final = {
    "team_id": "team-a",
    "team_alias": "Team A",
    "members_with_roles": [{"user_id": "old", "user_email": "old@corp.invalid", "role": "admin"}],
}


async def _add(cache: UserApiKeyCache | None = None) -> None:
    """Runs the step with a Prisma client that has no table, so any statement it sent through Prisma would fail."""
    user_api_key_cache: Final = cache or UserApiKeyCache()
    await add_team_member(
        SimpleNamespace(),  # pyright: ignore[reportArgumentType]  # the step must not reach Prisma
        user_api_key_cache,
        ProxyLogging(user_api_key_cache=user_api_key_cache),
        "default_user_id",
        "new",
        "team-a",
    )


@pytest.mark.asyncio
async def test_a_user_is_added_through_the_repository_and_the_rows_the_add_changed_leave_the_cache(
    rust_auth_stores: RustAuthStores,
) -> None:
    rust_auth_stores.team.rows["team-a"] = TEAM
    cache: Final = UserApiKeyCache()
    changed: Final = ("team_id:team-a", "team_alias:Team A", "new", "team_membership:new:team-a", "team-a_new")
    for key in (*changed, "bystander"):
        await cache.async_set_cache(key=key, value={"cached": key})

    await _add(cache)

    assert rust_auth_stores.team_membership.adds == [("new", "team-a", None)]
    assert [await cache.async_get_cache(key=key) for key in (*changed, "bystander")] == [
        *(None for _ in changed),
        {"cached": "bystander"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stored", "linked"), [({"team-default": {"budget_id": "team-default"}}, "team-default"), ({}, None)]
)
async def test_a_member_is_linked_to_the_teams_default_member_budget_while_that_budget_exists(
    rust_auth_stores: RustAuthStores, stored: dict[str, dict[str, object]], linked: str | None
) -> None:
    rust_auth_stores.team.rows["team-a"] = {**TEAM, "metadata": {"team_member_budget_id": "team-default"}}
    rust_auth_stores.budget.rows.update(stored)

    await _add()

    assert rust_auth_stores.team_membership.adds == [("new", "team-a", linked)]
    assert rust_auth_stores.team_membership.own_budgets == {}


@pytest.mark.asyncio
async def test_a_member_of_a_team_with_default_member_models_gets_its_own_budget_in_the_member_adds_transaction(
    rust_auth_stores: RustAuthStores,
) -> None:
    rust_auth_stores.team.rows["team-a"] = {
        **TEAM,
        "default_team_member_models": ["gpt-4o"],
        "metadata": {"team_member_budget_id": "team-default"},
    }
    before: Final = datetime.now(timezone.utc).timestamp()

    rust_auth_stores.budget.rows["team-default"] = {"budget_id": "team-default", "budget_duration": "30d"}

    await _add()

    ((user_id, team_id, budget_id),) = rust_auth_stores.team_membership.adds
    models, duration, reset_at, created_by, updated_by = rust_auth_stores.team_membership.own_budgets[str(budget_id)]
    assert (user_id, team_id, uuid.UUID(str(budget_id)).version) == ("new", "team-a", 4)
    assert (tuple(models), duration, created_by, updated_by) == (
        ("gpt-4o",),
        "30d",
        "default_user_id",
        "default_user_id",
    )
    assert reset_at is not None and reset_at > before


@pytest.mark.asyncio
async def test_an_add_that_does_not_land_keeps_the_cached_user_and_adds_no_budget(
    rust_auth_stores: RustAuthStores,
) -> None:
    rust_auth_stores.team.rows["team-a"] = {**TEAM, "default_team_member_models": ["gpt-4o"]}
    rust_auth_stores.team_membership.add_outcome = "already_member"
    cache: Final = UserApiKeyCache()
    await cache.async_set_cache(key="new", value={"cached": "user"})

    await _add(cache)

    assert rust_auth_stores.team_membership.own_budgets == {}
    assert await cache.async_get_cache(key="new") == {"cached": "user"}


@pytest.mark.asyncio
async def test_an_add_that_fails_raises_what_the_repository_raised(rust_auth_stores: RustAuthStores) -> None:
    rust_auth_stores.team.rows["team-a"] = {**TEAM, "default_team_member_models": ["gpt-4o"]}
    outage: Final = ConnectionError("database unreachable")
    rust_auth_stores.team_membership.add_error = outage

    with pytest.raises(ConnectionError) as raised:
        await _add()

    assert raised.value is outage


@pytest.mark.asyncio
async def test_a_user_the_roster_already_lists_is_not_added_again(rust_auth_stores: RustAuthStores) -> None:
    rust_auth_stores.team.rows["team-a"] = {
        **TEAM,
        "default_team_member_models": ["gpt-4o"],
        "members_with_roles": [{"user_id": "new", "role": "user"}],
    }
    await _add()

    assert (rust_auth_stores.team_membership.adds, rust_auth_stores.team_membership.own_budgets) == ([], {})


@pytest.mark.asyncio
@pytest.mark.parametrize(("outcome", "gone"), [("team_missing", "Team=team-a"), ("user_missing", "User=new")])
async def test_an_add_whose_team_or_user_was_deleted_meanwhile_is_refused_naming_what_is_gone(
    rust_auth_stores: RustAuthStores, outcome: str, gone: str
) -> None:
    rust_auth_stores.team.rows["team-a"] = TEAM
    rust_auth_stores.team_membership.add_outcome = outcome  # pyright: ignore[reportAttributeAccessIssue]  # parametrized over the literal's values

    with pytest.raises(HTTPException) as refused:
        await _add()

    assert (refused.value.status_code, refused.value.detail) == (
        404,
        {"error": f"{gone} was deleted while this member add was running"},
    )
