<template>
  <div class="compression_page">
    <h1 class="title">Compression</h1>
    <div class="comment">Upload facial image to compress</div>
    <div class="upload_area" ref="uploadArea" @click="selectFile">
      <div class="tips">
        <div>Click here to Select Image to upload</div>
        <div>or Drag & drop Image here 😊</div>
      </div>
      <input
        class="none"
        type="file"
        accept="image/*"
        @change="getFile"
        ref="fileInput"
        name="files"
      />
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      fileReader: null,
      fileData: [],
    };
  },
  created() {
    this.fileReader = new FileReader();
    this.fileReader.addEventListener('load', () => {
      this.fileData.push(this.fileReader.result);
      console.log(this.fileData);
    });
  },
  mounted() {
    const dropArea = this.$refs.uploadArea;
    dropArea.addEventListener('drop', this.dropEvent);
    dropArea.addEventListener('dragenter', this.preventFn);
    dropArea.addEventListener('dragover', this.preventFn);
  },
  methods: {
    selectFile() {
      this.$refs.fileInput.click();
    },
    addFiles(files) {
      console.log(files);
      files.forEach((file) => {
        this.fileReader.readAsDataURL(file);
      });
    },
    getFile(e) {
      this.addFiles(e.target.files);
    },
    dropEvent(e) {
      this.preventFn(e);
      this.addFiles(e.dataTransfer.files);
    },
    preventFn(e) {
      e.stopPropagation();
      e.preventDefault();
    },
  },
};
</script>

<style lang="scss" scoped>
.compression_page {
  margin: 0 auto;
  padding-top: 80px;
  width: 900px;

  .title {
    margin: 0;
    margin-bottom: 8px;
  }

  .comment {
    color: #949494;
    font-size: 20px;
    margin-bottom: 16px;
  }

  .upload_area {
    border: 1px #949494 dotted;
    border-radius: 8px;
    height: 450px;

    .tips {
      height: 100%;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      font-size: 32px;
      opacity: 0.3;

      & > div {
        margin-bottom: 12px;
      }
    }
  }
}
</style>
