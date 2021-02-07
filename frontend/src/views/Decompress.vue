<template>
  <section class="decompressor">
    <header class="title_wrapper">
      <h1 class="title">Decompressor</h1>
      <div class="comment">Upload compressed images to decompress</div>
    </header>
    <Uploader accept=".fic" />
  </section>
</template>

<script>
import Uploader from '@/components/Uploader.vue';

export default {
  components: { Uploader },
};
</script>

<style lang="scss" scoped>
.decompressor {
  margin: 0 auto;
  padding: 60px 0;
  width: 900px;

  .title {
    font-size: 32px;
    margin: 0;
    margin-bottom: 8px;
  }

  .comment {
    color: var(--text2);
    font-size: 20px;
    margin-bottom: 16px;
  }
}
</style>
