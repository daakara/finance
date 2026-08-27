import urllib.request
import re

def verify():
    print("Fetching Cloudflare Pages index HTML...")
    req = urllib.request.Request(
        "https://finance-xp8.pages.dev",
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode("utf-8")

    chunks = re.findall(r'/_next/static/chunks/[a-zA-Z0-9_\-\.]+\.js', html)
    print(f"Found {len(chunks)} JS chunks referenced in HTML.")

    railway_detected = False
    for chunk in set(chunks):
        chunk_url = f"https://finance-xp8.pages.dev{chunk}"
        try:
            req_chunk = urllib.request.Request(chunk_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req_chunk) as chunk_resp:
                content = chunk_resp.read().decode("utf-8")
                if "web-production-e370b.up.railway.app" in content:
                    print(f"  [CONFIRMED] Railway URL found in chunk: {chunk}")
                    railway_detected = True
                elif "onrender.com" in content:
                    print(f"  [NOTICE] Legacy Render reference in chunk: {chunk}")
        except Exception as e:
            print(f"  [ERROR] Could not fetch {chunk}: {e}")

    if railway_detected:
        print("\nSUCCESS: Cloudflare Pages is actively running with the Railway production backend URL!")
    else:
        print("\nNOTE: Railway URL not yet detected in live chunks. Cloudflare build may still be finishing or caching.")

if __name__ == "__main__":
    verify()
