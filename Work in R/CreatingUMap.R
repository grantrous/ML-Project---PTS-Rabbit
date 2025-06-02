#Merge 9 samples
allprostate <- merge(HT771, y = list(HT781, HT817, HT849, HT898, HT891,HT814,HT832,HT913), add.cell.ids = c('HT771','HT781','HT817','HT849','HT898','HT891','HT814','HT832','HT913'))

#Run PCA
allprostate <- NormalizeData(allprostate)
allprostate <- FindVariableFeatures(allprostate)
allprostate <- ScaleData(allprostate)
allprostate <- RunPCA(allprostate)

#Create UMAP
allprostate <- FindNeighbors(allprostate, dims = 1:25, graph.name = 'custom_snn')
allprostate <- FindClusters(allprostate, resolution = 1, graph.name = 'custom_snn') 
allprostate <- RunUMAP(allprostate, dims = 1:25)
DimPlot(allprostate, reduction = "umap", label = TRUE)

#Harmony Integration
allharmony <- allprostate
DefaultAssay(allharmony) <- 'RNA'
allharmony[["RNA"]] <- split(allharmony[["RNA"]], f = allharmony$orig.ident)
allharmony <- NormalizeData(allharmony)
allharmony <- FindVariableFeatures(allharmony, selection.method = "vst", nfeatures = 2000)
allharmony <- ScaleData(allharmony)
allharmony <- RunPCA(allharmony)
allharmony <- IntegrateLayers(  object = allharmony, method = HarmonyIntegration,  orig.reduction = "pca", new.reduction = "harmony",  verbose = FALSE)

#Recreate Umap with harmony
allharmony <- JoinLayers(allharmony)
allharmony <- RunUMAP(allharmony, dims = 1:30, reduction = 'harmony')
allharmony <- FindNeighbors(allharmony, dims = 1:30,reduction = 'harmony')
allharmony <- FindClusters(allharmony, resolution = 2, graph = 'RNA_snn')
DimPlot(allharmony, reduction = "umap", label = TRUE)
DimPlot(allharmony, reduction = "umap", label = TRUE, group.by = 'predicted_doublet')
allharmony <- subset(allharmony, subset = predicted_doublet == 'TRUE', invert = T)