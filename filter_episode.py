import json
import gzip

# Load original file
print("Loading original file...")
with gzip.open('data/datasets/objectnav/hm3d/v1/val/content/4ok3usBNeis.json.gz', 'rt') as f:
    data = json.load(f)

print(f"Original has {len(data['episodes'])} episodes")

# Find Episode 0
episode_0 = None
for ep in data['episodes']:
    if ep['episode_id'] == '0':
        episode_0 = ep
        break

if episode_0 is None:
    print("ERROR: Episode 0 not found!")
    exit(1)

print(f"Found Episode 0, target object: {episode_0['object_category']}")

# Replace episodes array with just Episode 0
data['episodes'] = [episode_0]

print(f"New episodes count: {len(data['episodes'])}")

# Save to test_episodes
print("Saving to test_episodes...")
with gzip.open('data/datasets/objectnav/hm3d/v1/test_episodes/content/4ok3usBNeis.json.gz', 'wt') as f:
    json.dump(data, f)

print("Done! Verifying...")

# Verify
with gzip.open('data/datasets/objectnav/hm3d/v1/test_episodes/content/4ok3usBNeis.json.gz', 'rt') as f:
    verify = json.load(f)
    print(f"Verification: {len(verify['episodes'])} episode(s)")
    print(f"Episode IDs: {[ep['episode_id'] for ep in verify['episodes']]}")
