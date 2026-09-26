from fastapi import APIRouter

router = APIRouter(tags=["health"])


# Explicitly declare both GET and HEAD. FastAPI/Starlette does NOT
# automatically add HEAD support to a GET-only route, so uptime pingers
# that send HEAD requests (e.g. UptimeRobot on its free tier, which
# doesn't allow choosing GET) get a 405 Method Not Allowed unless HEAD
# is registered here too.
@router.get("/health")
@router.head("/health")
async def health():
    return {"status": "ok"}
