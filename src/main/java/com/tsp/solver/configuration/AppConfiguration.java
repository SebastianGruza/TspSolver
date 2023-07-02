package com.tsp.solver.configuration;

import com.tsp.solver.data.Distances;
import com.tsp.solver.data.DistancesService;
import org.springframework.beans.factory.config.ConfigurableBeanFactory;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Scope;

import java.util.List;

@Configuration
@ConfigurationProperties(prefix = "tsp")
public class AppConfiguration {

    @Bean
    @Scope(ConfigurableBeanFactory.SCOPE_SINGLETON)
    public DistancesService distancesService() {
        DistancesService distancesService = new DistancesService();
        distancesService.updateDistances(new Distances(this));
        return distancesService;
    }

    private String filename;
    private int gpuThreads;
    private int colonyMultiplier;
    private int divideGreedy;
    private double scaleTime;
    private Boolean mergeColonyByTime;
    public List<Double> cutoffsByTime;

    public Boolean getMergeColonyByTime() {
        return mergeColonyByTime;
    }

    public void setMergeColonyByTime(Boolean mergeColonyByTime) {
        this.mergeColonyByTime = mergeColonyByTime;
    }

    public List<Double> getCutoffsByTime() {
        return cutoffsByTime;
    }

    public void setCutoffsByTime(List<Double> cutoffsByTime) {
        this.cutoffsByTime = cutoffsByTime;
    }

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

