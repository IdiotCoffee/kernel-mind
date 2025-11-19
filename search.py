
import chromadb

def pretty(results):
    for i in range(len(results["documents"][0])):
        print("\n=== Result", i+1, "===")
        print("Path   :", results["metadatas"][0][i]["path"])
        print("Name   :", results["metadatas"][0][i].get("name"))
        print("Type   :", results["metadatas"][0][i]["type"])
        print("Score  :", results["distances"][0][i])
        print("\nCode:\n")
        print(results["documents"][0][i])

def search(query, k=5):
    client = chromadb.PersistentClient(path=".chromadb")
    col = client.get_collection("kernelmind_index")

    res = col.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    pretty(res)
    return res
