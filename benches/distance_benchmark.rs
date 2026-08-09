use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use lightvector::distance::{cosine_distance, dot_product, l2_squared};
use rand::Rng;

fn generate_random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

fn bench_distance_metrics(c: &mut Criterion) {
    let dimensions = [384, 768, 1536];

    for dim in dimensions {
        let vec_a = generate_random_vector(dim);
        let vec_b = generate_random_vector(dim);

        let mut group = c.benchmark_group(format!("distance_{}_dim", dim));

        group.bench_function(BenchmarkId::new("L2_squared", dim), |b| {
            b.iter(|| l2_squared(black_box(&vec_a), black_box(&vec_b)))
        });

        group.bench_function(BenchmarkId::new("Cosine", dim), |b| {
            b.iter(|| cosine_distance(black_box(&vec_a), black_box(&vec_b)))
        });

        group.bench_function(BenchmarkId::new("DotProduct", dim), |b| {
            b.iter(|| dot_product(black_box(&vec_a), black_box(&vec_b)))
        });

        group.finish();
    }
}

criterion_group!(benches, bench_distance_metrics);
criterion_main!(benches);
