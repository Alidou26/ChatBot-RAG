import os
import time
import requests
from dotenv import load_dotenv


# ==========================================================
#                 Redmine API Data Fetcher
# ==========================================================
# This module connects to a Redmine server using its REST API.
# It retrieves *all* available information for a given project:
#   - Project details (metadata)
#   - Issues (with attachments & journals)
#   - Versions (milestones)
#   - Memberships (users & roles)
#   - News
#   - Issue categories
#   - Files uploaded to the project
#   - Wiki pages (including attachments)
#
# The returned data can be used for analysis, backups, or feeding
# into an LLM. It does NOT download binary files; that’s handled
# separately in main.py.
# ==========================================================


# ----------------------------------------------------------
#   Load configuration from .env
# ----------------------------------------------------------
# Example .env file:
#   REDMINE_URL=https://redmine.example.com
#   REDMINE_API_KEY=abcdef123456789
#   REDMINE_PROJECT_ID=projet-drone-terrestre-autonome
load_dotenv()


BASE = os.environ["REDMINE_URL"]
API_KEY = os.environ["REDMINE_API_KEY"]
PROJECT_ID = os.environ.get("REDMINE_PROJECT_ID") 
HEADERS = {"X-Redmine-API-Key": API_KEY}




# ----------------------------------------------------------
#   Helper to send GET requests safely
# ----------------------------------------------------------
def get_json(url, params=None):
    """
    Sends a GET request to the specified Redmine API endpoint
    and returns the parsed JSON response.


    Args:
        url (str): API endpoint.
        params (dict): Optional query parameters.


    Returns:
        dict: JSON-decoded response.
    """
    resp = requests.get(url, headers=HEADERS, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()




# ----------------------------------------------------------
#   Handle pagination (Redmine returns 100 items max/page)
# ----------------------------------------------------------
def fetch_all_paginated(endpoint, key):
    """
    Retrieves *all* paginated data from a Redmine API endpoint.


    Args:
        endpoint (str): The API endpoint (e.g. /issues.json).
        key (str): The JSON key containing the data list (e.g. "issues").


    Returns:
        list: Combined list of all items across all pages.
    """
    offset, limit = 0, 100
    items = []


    while True:
        data = get_json(endpoint, {"limit": limit, "offset": offset})
        chunk = data.get(key, [])
        items.extend(chunk)


        total = data.get("total_count", len(items))
        print(f"  ↳ Fetched {len(items)}/{total} {key}")


        if len(chunk) == 0 or offset >= total:
            break


        offset += limit
        time.sleep(0.2)  # be polite to server


    return items




# ----------------------------------------------------------
#   Main function: fetch full Redmine project data
# ----------------------------------------------------------
def fetch_full_project_data(project_id, status_id="open"):
    """
    Fetches absolutely everything about a Redmine project.


    Args:
        project_id (str|int): Redmine project identifier or ID.
        status_id (str): Filter for issues ("open", "closed", "*").


    Returns:
        dict: All project data combined.
    """


    # --- Project Metadata ---
    print(f"\n Fetching project info for '{project_id}' ...")
    project_url = f"{BASE}/projects/{project_id}.json"
    project = get_json(project_url).get("project", {})


    # --- Issues (with attachments & journals) ---
    print(" Fetching issues ...")
    issues_endpoint = (
        f"{BASE}/issues.json?"
        f"project_id={project_id}&status_id={status_id}&include=attachments,journals"
    )
    issues = fetch_all_paginated(issues_endpoint, "issues")


    # --- Versions (Milestones) ---
    print(" Fetching versions ...")
    versions = fetch_all_paginated(f"{BASE}/projects/{project_id}/versions.json", "versions")


    # --- Memberships (Users & Roles) ---
    print(" Fetching memberships ...")
    memberships = fetch_all_paginated(f"{BASE}/projects/{project_id}/memberships.json", "memberships")


    # --- News ---
    print(" Fetching news ...")
    news = fetch_all_paginated(f"{BASE}/projects/{project_id}/news.json", "news")


    # --- Issue Categories ---
    print(" Fetching issue categories ...")
    issue_categories = fetch_all_paginated(
        f"{BASE}/projects/{project_id}/issue_categories.json",
        "issue_categories"
    )


    # --- Files uploaded directly to the project ---
    print(" Fetching project files ...")
    files = fetch_all_paginated(f"{BASE}/projects/{project_id}/files.json", "files")


    # --- Wiki Pages (text + attachments) ---
    print(" Fetching wiki pages ...")
    wiki_index_url = f"{BASE}/projects/{project_id}/wiki/index.json"
    wiki_index = get_json(wiki_index_url).get("wiki_pages", [])
    wiki_pages = []


    for page in wiki_index:
        title = page["title"]
        try:
        # Fetch full wiki page with attachments included
            page_url = f"{BASE}/projects/{project_id}/wiki/{title}.json?include=attachments"
            full_page = get_json(page_url)
            wiki_page = full_page.get("wiki_page", {})
        
        # Debugging info — see what was found
            attachments = wiki_page.get("attachments", [])
            if attachments:
                print(f"Found {len(attachments)} attachments on wiki page '{title}'")
            else:
                print(f"No attachments on wiki page '{title}'")
        
            wiki_pages.append(wiki_page)
            time.sleep(0.2)
        except Exception as e:
            print(f"Failed to fetch wiki page '{title}': {e}")



    # --- Return everything together ---
    return {
        "project": project,
        "issues": issues,
        "versions": versions,
        "memberships": memberships,
        "news": news,
        "issue_categories": issue_categories,
        "files": files,
        "wiki": wiki_pages,
        "total_issues": len(issues)
    }




# ----------------------------------------------------------
#  Manual test execution
# ----------------------------------------------------------
if __name__ == "__main__":
    data = fetch_full_project_data(project_id=PROJECT_ID, status_id="open")


    print("\n Fetch complete!")
    print(f"Project: {data['project'].get('name')}")
    print(f"Issues: {data['total_issues']}")
    print(f"Wiki pages: {len(data['wiki'])}")
    print(f"Versions: {len(data['versions'])}")
    print(f"Members: {len(data['memberships'])}")