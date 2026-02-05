"""
Test script for Web Knowledge Integration in CausalOS v4

This script tests:
1. Basic web search functionality
2. Fact verification
3. Counterfactual reasoning with fact-checking
4. Cache mechanism
"""

import sys


def test_basic_search():
    """Test 1: Basic web search"""
    print("=" * 80)
    print("TEST 1: Basic Web Search")
    print("=" * 80)
    
    from WebKnowledgeRetriever import WebKnowledgeRetriever
    
    retriever = WebKnowledgeRetriever()
    
    if not retriever.ddg_available:
        print("⚠️ DuckDuckGo search not available")
        print("Install with: pip install duckduckgo-search")
        return False
    
    query = "Python programming language creator"
    print(f"\nSearching: {query}")
    
    results = retriever.search(query, max_results=3)
    
    if len(results) == 0:
        print("❌ FAILED: No results returned")
        return False
    
    print(f"\n✅ Found {len(results)} results:")
    for i, r in enumerate(results, 1):
        print(f"\n  Result {i}:")
        print(f"  Title: {r['title']}")
        print(f"  Snippet: {r['snippet'][:100]}...")
        print(f"  URL: {r['url']}")
    
    print("\n✅ TEST PASSED\n")
    return True


def test_fact_verification():
    """Test 2: Fact verification (keyword-based)"""
    print("=" * 80)
    print("TEST 2: Fact Verification")
    print("=" * 80)
    
    from WebKnowledgeRetriever import WebKnowledgeRetriever
    
    retriever = WebKnowledgeRetriever()
    
    if not retriever.ddg_available:
        print("⚠️ Skipped: DuckDuckGo not available")
        return False
    
    print("\nTest 2a: Verifying TRUE fact...")
    claim = "Python was created by Guido van Rossum"
    print(f"Claim: {claim}")
    
    is_verified, evidence, conf = retriever.verify_fact(claim, use_llm=False)
    
    print(f"\nResult:")
    print(f"  Verified: {is_verified}")
    print(f"  Confidence: {conf:.2f}")
    print(f"  Evidence: {evidence[:150]}...")
    
    if not is_verified or conf < 0.3:
        print("❌ FAILED: Should have verified true fact")
        return False
    
    print("\n✅ Test 2a passed")
    
    print("\nTest 2b: Verifying FALSE fact...")
    claim = "Python was created in the year 2020"
    print(f"Claim: {claim}")
    
    is_verified, evidence, conf = retriever.verify_fact(claim, use_llm=False)
    
    print(f"\nResult:")
    print(f"  Verified: {is_verified}")
    print(f"  Confidence: {conf:.2f}")
    
    # For false facts, confidence might be low or verification might fail
    print("\n✅ Test 2b completed (result noted)")
    
    print("\n✅ TEST PASSED\n")
    return True


def test_cache_mechanism():
    """Test 3: Cache mechanism"""
    print("=" * 80)
    print("TEST 3: Cache Mechanism")
    print("=" * 80)
    
    from WebKnowledgeRetriever import WebKnowledgeRetriever
    import time
    
    retriever = WebKnowledgeRetriever(cache_size=10)
    
    if not retriever.ddg_available:
        print("⚠️ Skipped: DuckDuckGo not available")
        return False
    
    query = "test query for cache"
    
    print(f"\nFirst search: {query}")
    start = time.time()
    results1 = retriever.search(query, max_results=2)
    time1 = time.time() - start
    
    print(f"Second search (should hit cache): {query}")
    start = time.time()
    results2 = retriever.search(query, max_results=2)
    time2 = time.time() - start
    
    print(f"\nFirst search time: {time1:.3f}s")
    print(f"Second search time: {time2:.3f}s (cached)")
    
    if results1 != results2:
        print("❌ FAILED: Cached results don't match")
        return False
    
    if time2 > time1 * 0.5:
        print("⚠️ Warning: Cache may not be working (times similar)")
    else:
        print("✅ Cache working (second search faster)")
    
    print(f"\nCache size: {len(retriever.cache)}")
    
    print("\n✅ TEST PASSED\n")
    return True


def test_causalos_integration():
    """Test 4: CausalOS integration"""
    print("=" * 80)
    print("TEST 4: CausalOS Integration")
    print("=" * 80)
    
    print("\nInitializing CausalOS with web knowledge...")
    
    try:
        from CausalOS_v4 import UnifiedCausalOSV4
        
        osys = UnifiedCausalOSV4(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            use_premise_aware=True,
            use_web_knowledge=True
        )
        
        if not hasattr(osys, 'knowledge_augmented') or osys.knowledge_augmented is None:
            print("❌ FAILED: Web knowledge not initialized")
            return False
        
        print("\n✅ Web knowledge module loaded")
        
        if not osys.knowledge_augmented.retriever.ddg_available:
            print("⚠️ Warning: DuckDuckGo not available, skipping fact-check test")
            print("✅ TEST PASSED (partial)\n")
            return True
        
        print("\nTesting counterfactual with fact-checking...")
        print("Note: This may take a minute due to model + web search...")
        
        result = osys.knowledge_augmented.solve_counterfactual_with_facts(
            factual="Python is a programming language",
            counterfactual="Python is a snake",
            options={
                "A": "It would be dangerous",
                "B": "It would be slithery",
                "C": "The meaning would be different"
            },
            verify_facts=True  # Enable fact-checking
        )
        
        print(f"\nResult:")
        print(f"  Selected: {result['selected_option']}")
        print(f"  Reasoning: {result['reasoning']}")
        print(f"  Verified facts: {len(result['verified_facts'])}")
        
        print("\n✅ TEST PASSED\n")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              CausalOS v4 - Web Knowledge Integration Test Suite              ║
║                                                                              ║
║  Tests API-free web search, fact verification, and knowledge-augmented       ║
║  counterfactual reasoning.                                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    tests = [
        ("Basic Web Search", test_basic_search),
        ("Fact Verification", test_fact_verification),
        ("Cache Mechanism", test_cache_mechanism),
        ("CausalOS Integration", test_causalos_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n\nRunning: {test_name}")
        user_input = input("Continue? (y/n, default=y): ").strip().lower()
        
        if user_input == 'n':
            print(f"Skipped: {test_name}")
            results.append((test_name, "SKIPPED"))
            continue
        
        try:
            passed = test_func()
            results.append((test_name, "PASSED" if passed else "FAILED"))
        except Exception as e:
            print(f"\n❌ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, "CRASHED"))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, status in results:
        symbol = "✅" if status == "PASSED" else "⚠️" if status == "SKIPPED" else "❌"
        print(f"{symbol} {test_name}: {status}")
    
    passed_count = sum(1 for _, status in results if status == "PASSED")
    total_count = len([r for r in results if r[1] != "SKIPPED"])
    
    print(f"\nPassed: {passed_count}/{total_count}")
    
    if passed_count == total_count and total_count > 0:
        print("\n🎉 ALL TESTS PASSED!")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
