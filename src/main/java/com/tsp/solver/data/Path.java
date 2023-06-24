package com.tsp.solver.data;


import java.util.Objects;

public class Path {
    private double total;

    public Double getTotal() {
        return total;
    }

    public Path(double total) {
        this.total = total;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Path path = (Path) o;
        return path.total >= total - 0.0000001 && path.total <= total + 0.0000001;
    }

    @Override
    public int hashCode() {
        return Objects.hash((int) total);
    }

    @Override
    public String toString() {
        return String.format("%.6f", total);
    }
}
