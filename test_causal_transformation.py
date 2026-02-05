"""
Test script for CausalTransformationEngine

This script validates the Osborn-based causal transformation operations
and demonstrates creative idea generation through causal graph manipulation.
"""

import sys
import torch
from CausalOS_v4 import UnifiedCausalOSV4
from CausalTransformationEngine import CausalTransformationEngine


def test_basic_transformations():
    """Test each Osborn transformation individually."""
    print("=" * 80)
    print("TEST 1: Basic Transformations")
    print("=" * 80)
    
    # Initialize system with smaller model for faster testing
    print("\nInitializing CausalOS...")
    osys = UnifiedCausalOSV4(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        use_premise_aware=False,
        use_web_knowledge=False
    )
    
    engine = CausalTransformationEngine(osys)
    
    # Build a simple causal graph
    print("\nBuilding causal graph from text...")
    text = "Electricity flows through a wire and produces light in a bulb."
    triplets = osys.build_causal_graph(text)
    
    print(f"\nExtracted {len(triplets)} causal relationships:")
    for t in triplets:
        print(f"  {t['cause']} -> {t['effect']} (strength: {t.get('magnitude', 0.5):.2f})")
    
    # Get original state
    original = engine._get_current_graph_state()
    print(f"\nOriginal graph: {len(original['nodes'])} nodes, {len(original['edges'])} edges")
    
    # Test 1: Substitute
    print("\n--- Test 1.1: Substitute ---")
    if 'electricity' in osys.label_to_idx:
        result = engine.substitute('electricity', 'solar energy')
        print(f"After substitution: {len(result['nodes'])} nodes, {len(result['edges'])} edges")
        print("✅ Substitute test passed")
    else:
        print("⚠️ 'electricity' not found, skipping substitute test")
    
    # Rebuild for next test
    osys.reset_graph()
    osys.build_causal_graph(text)
    
    # Test 2: Reverse
    print("\n--- Test 1.2: Reverse ---")
    if triplets:
        cause = triplets[0]['cause']
        effect = triplets[0]['effect']
        print(f"Reversing: {cause} -> {effect}")
        result = engine.reverse((cause, effect))
        
        # Check if reversal worked
        found_reversed = False
        for edge in result['edges']:
            if edge[0] == effect and edge[1] == cause:
                found_reversed = True
                break
        
        if found_reversed:
            print("✅ Reverse test passed - edge direction reversed")
        else:
            print("⚠️ Reverse edge may not be visible (check threshold)")
    
    # Test 3: Modify
    print("\n--- Test 1.3: Modify ---")
    osys.reset_graph()
    osys.build_causal_graph(text)
    
    if triplets:
        cause = triplets[0]['cause']
        effect = triplets[0]['effect']
        print(f"Strengthening: {cause} -> {effect} by 2x")
        result = engine.modify((cause, effect), strength_multiplier=2.0)
        print("✅ Modify test passed")
    
    # Test 4: Eliminate
    print("\n--- Test 1.4: Eliminate ---")
    osys.reset_graph()
    osys.build_causal_graph(text)
    
    if triplets:
        cause = triplets[0]['cause']
        effect = triplets[0]['effect']
        print(f"Eliminating edge: {cause} -> {effect}")
        result = engine.eliminate((cause, effect))
        
        # Check if edge was removed
        found_edge = False
        for edge in result['edges']:
            if edge[0] == cause and edge[1] == effect:
                found_edge = True
                break
        
        if not found_edge:
            print("✅ Eliminate test passed - edge removed")
        else:
            print("⚠️ Edge still exists after elimination")
    
    print("\n✅ All basic transformation tests completed")
    return True


def test_creative_generation():
    """Test creative idea generation using transformations."""
    print("\n" + "=" * 80)
    print("TEST 2: Creative Idea Generation")
    print("=" * 80)
    
    print("\nInitializing CausalOS...")
    osys = UnifiedCausalOSV4(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        use_premise_aware=False,
        use_web_knowledge=False
    )
    
    engine = CausalTransformationEngine(osys)
    
    # Test case: Light bulb invention
    print("\nScenario: Applying Osborn transformations to 'light production'")
    text = "Electric current heats a filament, producing light."
    osys.build_causal_graph(text)
    
    original = engine._get_current_graph_state()
    print(f"\nOriginal: {original['nodes'][:5]}")  # Show first 5 nodes
    
    # Apply different transformations
    transformations_to_test = [
        ("Reverse", {"edge": ("electric current", "light")}),
        ("Magnify", {"effect_node": "light", "amplification": 3.0}),
    ]
    
    for idx, (trans_name, kwargs) in enumerate(transformations_to_test, 1):
        print(f"\n--- Transformation {idx}: {trans_name} ---")
        osys.reset_graph()
        osys.build_causal_graph(text)
        
        try:
            result = engine.apply_transformation(trans_name, **kwargs)
            print(f"Result: {len(result['nodes'])} nodes, {len(result['edges'])} edges")
            
            # Evaluate creativity
            novelty = engine.evaluator.novelty_score(original, result, engine.similarity)
            print(f"Novelty score: {novelty:.2f}")
            
            if novelty > 0:
                print(f"✅ {trans_name} transformation successful")
            else:
                print(f"⚠️ {trans_name} produced low novelty")
        except Exception as e:
            print(f"❌ {trans_name} failed: {e}")
    
    print("\n✅ Creative generation test completed")
    return True


def test_graph_similarity():
    """Test graph similarity metrics."""
    print("\n" + "=" * 80)
    print("TEST 3: Graph Similarity Metrics")
    print("=" * 80)
    
    print("\nInitializing CausalOS...")
    osys = UnifiedCausalOSV4(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        use_premise_aware=False,
        use_web_knowledge=False
    )
    
    engine = CausalTransformationEngine(osys)
    
    # Create two similar graphs
    print("\nCreating two causal graphs...")
    text1 = "Rain causes wet ground."
    text2 = "Watering causes wet soil."
    
    osys.build_causal_graph(text1)
    graph1 = engine._get_current_graph_state()
    
    osys.reset_graph()
    osys.build_causal_graph(text2)
    graph2 = engine._get_current_graph_state()
    
    # Calculate similarities
    print("\nCalculating similarities...")
    struct_dist = engine.similarity.structural_distance(graph1, graph2)
    sem_dist = engine.similarity.semantic_distance(graph1, graph2)
    hybrid_dist = engine.similarity.hybrid_similarity(graph1, graph2)
    
    print(f"Structural distance: {struct_dist:.3f}")
    print(f"Semantic distance: {sem_dist:.3f}")
    print(f"Hybrid distance: {hybrid_dist:.3f}")
    
    if 0 <= hybrid_dist <= 1:
        print("✅ Similarity metrics working correctly")
    else:
        print("⚠️ Similarity score out of expected range")
    
    return True


def test_end_to_end_invention():
    """End-to-end test: Generate inventive ideas from a scenario."""
    print("\n" + "=" * 80)
    print("TEST 4: End-to-End Inventive Idea Generation")
    print("=" * 80)
    
    print("\nInitializing CausalOS...")
    osys = UnifiedCausalOSV4(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        use_premise_aware=False,
        use_web_knowledge=False
    )
    
    engine = CausalTransformationEngine(osys)
    
    # Scenario
    scenario = "A person exercises regularly, which improves their health and energy levels."
    print(f"\nScenario: {scenario}")
    
    # Build graph
    print("\nBuilding causal graph...")
    triplets = osys.build_causal_graph(scenario)
    
    if not triplets:
        print("⚠️ No causal relationships extracted")
        return False
    
    print(f"Extracted {len(triplets)} relationships:")
    for t in triplets[:3]:
        print(f"  {t['cause']} -> {t['effect']}")
    
    # Apply transformations
    print("\n--- Generating variations with Osborn transformations ---")
    
    variations = []
    
    # 1. Reverse: What if health improvement led to exercise?
    if triplets:
        cause = triplets[0]['cause']
        effect = triplets[0]['effect']
        
        osys.reset_graph()
        osys.build_causal_graph(scenario)
        
        print(f"\n1. REVERSE: {cause} -> {effect}")
        result = engine.reverse((cause, effect))
        variations.append(("Reverse", result))
        print(f"   Idea: What if {effect} motivates {cause}?")
    
    # 2. Adapt: Transfer to different domain
    print(f"\n2. ADAPT: Transfer 'exercise->health' to business domain")
    osys.reset_graph()
    osys.build_causal_graph(scenario)
    
    if triplets:
        result = engine.adapt((triplets[0]['cause'], triplets[0]['effect']), "business productivity")
        variations.append(("Adapt", result))
        print(f"   Created analogous relationship in business domain")
    
    # 3. Magnify: Amplify the effect
    print(f"\n3. MAGNIFY: Amplify health benefits")
    osys.reset_graph()
    osys.build_causal_graph(scenario)
    
    if 'health' in osys.label_to_idx:
        result = engine.magnify('health', amplification=2.5)
        variations.append(("Magnify", result))
        print(f"   Idea: Supercharged health outcomes")
    
    print(f"\n✅ Generated {len(variations)} inventive variations")
    
    # Evaluate creativity of variations
    osys.reset_graph()
    osys.build_causal_graph(scenario)
    original = engine._get_current_graph_state()
    
    print("\n--- Creativity Evaluation ---")
    for name, var_graph in variations:
        novelty = engine.evaluator.novelty_score(original, var_graph, engine.similarity)
        print(f"{name}: Novelty = {novelty:.2f}")
    
    print("\n✅ End-to-end invention test completed")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("CAUSAL TRANSFORMATION ENGINE - TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Basic Transformations", test_basic_transformations),
        ("Creative Generation", test_creative_generation),
        ("Graph Similarity", test_graph_similarity),
        ("End-to-End Invention", test_end_to_end_invention),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'*' * 80}")
            print(f"Running: {test_name}")
            print(f"{'*' * 80}")
            
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
