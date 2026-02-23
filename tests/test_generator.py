"""Tests for evalflow.data.generator."""
import pytest

from evalflow.data.generator import DOMAINS, EDGE_CASES, DatasetGenerator


class TestDatasetGenerator:
    def test_generates_correct_size(self):
        gen = DatasetGenerator(seed=42)
        scenarios = gen.generate_synthetic_dataset(size=20)
        assert len(scenarios) == 20

    def test_deterministic_with_seed(self):
        gen1 = DatasetGenerator(seed=123)
        gen2 = DatasetGenerator(seed=123)
        s1 = gen1.generate_synthetic_dataset(size=10)
        s2 = gen2.generate_synthetic_dataset(size=10)
        assert [s.name for s in s1] == [s.name for s in s2]

    def test_includes_edge_cases(self):
        gen = DatasetGenerator(seed=42)
        scenarios = gen.generate_synthetic_dataset(size=50, include_edge_cases=True)
        categories = {s.category for s in scenarios}
        assert "adversarial" in categories or "safety" in categories or "edge_case" in categories

    def test_excludes_edge_cases(self):
        gen = DatasetGenerator(seed=42)
        scenarios = gen.generate_synthetic_dataset(size=20, include_edge_cases=False)
        for s in scenarios:
            assert s.domain != "edge_case"

    def test_filter_by_domain(self):
        gen = DatasetGenerator(seed=42)
        scenarios = gen.generate_synthetic_dataset(size=10, domains=["finance"], include_edge_cases=False)
        for s in scenarios:
            assert s.domain == "finance"

    def test_all_scenarios_have_required_fields(self):
        gen = DatasetGenerator(seed=42)
        scenarios = gen.generate_synthetic_dataset(size=50)
        for s in scenarios:
            assert s.id
            assert s.name
            assert s.description
            assert s.initial_context is not None  # Can be empty string for edge cases
            assert s.expected_tool_sequence is not None

    def test_domain_coverage(self):
        gen = DatasetGenerator(seed=42)
        scenarios = gen.generate_synthetic_dataset(size=100)
        domains = {s.domain for s in scenarios}
        # Should have at least a few domains
        assert len(domains) >= 3

    def test_difficulty_values(self):
        gen = DatasetGenerator(seed=42)
        scenarios = gen.generate_synthetic_dataset(size=50)
        for s in scenarios:
            assert s.difficulty in ("easy", "medium", "hard")


class TestDomainCatalogue:
    def test_all_domains_have_required_keys(self):
        for domain_name, domain in DOMAINS.items():
            assert "verbs" in domain, f"{domain_name} missing verbs"
            assert "subjects" in domain, f"{domain_name} missing subjects"
            assert "multi_step_triggers" in domain, f"{domain_name} missing triggers"
            assert len(domain["verbs"]) > 0
            assert len(domain["subjects"]) > 0

    def test_edge_cases_well_formed(self):
        for ec in EDGE_CASES:
            assert "name" in ec
            assert "description" in ec
            assert "initial_context" in ec
            assert "expected_tools" in ec
            assert "difficulty" in ec
            assert ec["difficulty"] in ("easy", "medium", "hard")


class TestDatasetHash:
    def test_hash_deterministic(self):
        from evalflow.data.generator import DatasetGenerator
        gen = DatasetGenerator(seed=42)
        ds1 = gen.generate_synthetic_dataset(size=10)
        gen2 = DatasetGenerator(seed=42)
        ds2 = gen2.generate_synthetic_dataset(size=10)
        assert gen.compute_dataset_hash(ds1) == gen2.compute_dataset_hash(ds2)

    def test_different_data_different_hash(self):
        from evalflow.data.generator import DatasetGenerator
        gen1 = DatasetGenerator(seed=42)
        gen2 = DatasetGenerator(seed=99)
        ds1 = gen1.generate_synthetic_dataset(size=10)
        ds2 = gen2.generate_synthetic_dataset(size=10)
        assert gen1.compute_dataset_hash(ds1) != gen2.compute_dataset_hash(ds2)


class TestEdgeCaseSQLInjection:
    def test_sql_injection_scenario_exists(self):
        from evalflow.data.generator import EDGE_CASES
        names = [e['name'] for e in EDGE_CASES]
        assert 'SQL injection attempt' in names
