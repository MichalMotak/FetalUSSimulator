﻿using UnityEngine;
using UnityEditor;
using System.IO;
using System;

namespace UnityVolumeRendering
{
    /// <summary>
    /// Editor window for importing datasets.
    /// </summary>
    public class RAWDatasetImporterEditorWindow : EditorWindow
    {
        private string fileToImport;

        private int dimX  = 249;
        private int dimY = 199;
        private int dimZ = 160;
        private int bytesToSkip = 0;
        private DataContentFormat dataFormat = DataContentFormat.Int16;
        private Endianness endianness = Endianness.LittleEndian;

        public RAWDatasetImporterEditorWindow(string filePath)
        {
            fileToImport = filePath;

            // if (Path.GetExtension(fileToImport) == ".ini")
            //     fileToImport = fileToImport.Replace(".ini", ".raw");

            // Try parse ini file (if available)
            DatasetIniData initData = DatasetIniReader.ParseIniFile(fileToImport + ".ini");
            if (initData != null)
            {
                Debug.Log("nii" + fileToImport);
                dimX = initData.dimX;
                dimY = initData.dimY;
                dimZ = initData.dimZ;
                bytesToSkip = initData.bytesToSkip;
                dataFormat = initData.format;
                endianness = initData.endianness;
            }

            this.minSize = new Vector2(300.0f, 200.0f);
        }

        private void ImportDataset()
        {
            RawDatasetImporter importer = new RawDatasetImporter(fileToImport, dimX, dimY, dimZ, dataFormat, endianness, bytesToSkip);
            VolumeDataset dataset = importer.Import();

            if (dataset != null)
            {
                if (EditorPrefs.GetBool("DownscaleDatasetPrompt"))
                {
                    if (EditorUtility.DisplayDialog("Optional DownScaling",
                        $"Do you want to downscale the dataset? The dataset's dimension is: {dataset.dimX} x {dataset.dimY} x {dataset.dimZ}", "Yes", "No"))
                    {
                        dataset.DownScaleData();
                    }
                }
                VolumeRenderedObject obj = VolumeObjectFactory.CreateObject(dataset);
            }
            else
            {
                Debug.LogError("Failed to import datset");
            }

            this.Close();
        }

        private void OnGUI()
        {
            dimX = EditorGUILayout.IntField("X dimension", dimX);
            dimY = EditorGUILayout.IntField("Y dimension", dimY);
            dimZ = EditorGUILayout.IntField("Z dimension", dimZ);
            bytesToSkip = EditorGUILayout.IntField("Bytes to skip", bytesToSkip);
            dataFormat = (DataContentFormat)EditorGUILayout.EnumPopup("Data format", dataFormat);
            endianness = (Endianness)EditorGUILayout.EnumPopup("Endianness", endianness);

            if (GUILayout.Button("Import"))
                ImportDataset();

            if (GUILayout.Button("Cancel"))
                this.Close();
        }
    }
}