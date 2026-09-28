#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TABLE_SIZE 64

typedef struct Entry {
    char *key;
    int value;
    struct Entry *next;
} Entry;

typedef struct { Entry *buckets[TABLE_SIZE]; } HashTable;

static unsigned int hash(const char *key) {
    unsigned int h = 5381;
    while (*key) h = ((h << 5) + h) + (unsigned char)*key++;
    return h % TABLE_SIZE;
}

HashTable *ht_create(void) { return calloc(1, sizeof(HashTable)); }

void ht_set(HashTable *ht, const char *key, int value) {
    unsigned int idx = hash(key);
    for (Entry *e = ht->buckets[idx]; e; e = e->next)
        if (strcmp(e->key, key) == 0) { e->value = value; return; }
    Entry *e = malloc(sizeof(Entry));
    e->key = strdup(key); e->value = value;
    e->next = ht->buckets[idx]; ht->buckets[idx] = e;
}

int ht_get(HashTable *ht, const char *key, int *out) {
    for (Entry *e = ht->buckets[hash(key)]; e; e = e->next)
        if (strcmp(e->key, key) == 0) { *out = e->value; return 1; }
    return 0;
}

int main(void) {
    HashTable *ht = ht_create();
    ht_set(ht, "alice", 42);
    ht_set(ht, "bob", 17);
    int val;
    if (ht_get(ht, "bob", &val)) printf("bob -> %d\n", val);
    if (!ht_get(ht, "dave", &val)) printf("dave -> not found\n");
    return 0;
}
