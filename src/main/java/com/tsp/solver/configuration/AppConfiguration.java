package com.tsp.solver.configuration;

import com.tsp.solver.data.Distances;
import org.springframework.beans.factory.config.ConfigurableBeanFactory;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Scope;

@Configuration
@ConfigurationProperties(prefix = "tsp")
public class AppConfiguration {

    @Bean
    @Scope(ConfigurableBeanFactory.SCOPE_SINGLETON)
    public Distances dist() {
        return new Distances(this);
    }

    private String filename;
    private int gpuThreads;
    private int colonyMultiplier;
    private int divideGreedy;

    private double scaleTime;

    public double getScaleTime() {
        return scaleTime;
    }

    public void setScaleTime(double scaleTime) {
        this.scaleTime = scaleTime;
    }

    public int getGpuThreads() {
        return gpuThreads;
    }
    public void setGpuThreads(int gpuThreads) {
        this.gpuThreads = gpuThreads;
    }
    public int getColonyMultiplier() {
        return colonyMultiplier;
    }
    public void setColonyMultiplier(int colonyMultiplier) {
        this.colonyMultiplier = colonyMultiplier;
    }

    public int getDivideGreedy() {
        return divideGreedy;
    }

    public void setDivideGreedy(int divideGreedy) {
        this.divideGreedy = divideGreedy;
    }
    public String getFilename() {
        return filename;
    }
    public void setFilename(String filename) {
        this.filename = filename;
    }

}

