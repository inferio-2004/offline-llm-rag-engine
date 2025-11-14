from pathlib import Path
from pybot_module import pyBot  # adjust import to your package name

# 1) Build a fresh instance
bot = pyBot(
    instance_name="test_bot_quick",
    llm_name="openhermes2b",
    save_dir="D:/.yourchatbot",
    force_rebuild=True,    # force clean build
    update=False,
    target_url="https://getbootstrap.com/"  # pick any small static site
)

# 2) Check that files exist on disk
base = Path("D:/.yourchatbot").expanduser() / "test_bot_quick"
assert (base / "chunks.db").exists(), "SQLite DB missing"
assert (base / "embeddings.npy").exists(), "Embeddings missing"
assert (base / "faiss.index").exists(), "FAISS index missing"
assert (base / "models" / "OpenHermes-Gemma-2B.Q8_0.gguf").exists(), "LLM file missing"
print("✅ Build artifacts are all present")

# 3) Test set_llm persistence
bot.set_llm(friendliness=7, ooc="No spoilers please", contact={"email":"bot@example.com"})
# Re-load settings from disk by creating a fresh instance loader
bot2 = pyBot(
    instance_name="test_bot_quick",
    llm_name="openhermes2b",
    save_dir="~/.yourchatbot",
    force_rebuild=False,
    update=False,
    target_url="https://getbootstrap.com/"
)
bot2.set_llm(friendliness=7, ooc="No spoilers please", contact={"email":"bot@example.com"})
assert bot2.llm_settings["friendliness"] == 7
assert bot2.llm_settings["ooc"] == "No spoilers please"
assert bot2.llm_settings["contact"]["email"] == "bot@example.com"
print("✅ set_llm() persisted correctly")

# 4) Test converse (should return a string)
query=''
while query!='\quit':
    query=input("enter ur query(type \quit to stop):")
    reply = bot2.converse(query)
    print("🔹 converse() replied:", reply)

# 5) Finally, test destroy
bot2.destroy()
assert not base.exists(), "Instance folder was not removed"
print("✅ destroy() cleaned everything up")
